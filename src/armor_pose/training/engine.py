"""Train or validate a registered backbone with the independent 4/9/8 armor head."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

from ..common.runtime import CLASS_NAMES, COLOR_NAMES, DetectionMetrics, decode_predictions, load_model

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from ..data.dataset import ArmorPoseDataset, armor_collate
from ..common.loss import ArmorPoseLoss
from ..models import build_model, MODEL_ALIASES, DEFAULT_MODEL
from ..settings import DEFAULT_TRAIN_CONFIG

DEFAULT_CONFIG = DEFAULT_TRAIN_CONFIG


def limited_dataset(dataset, maximum):
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    # Small smoke runs include both foreground and empty images; full runs use
    # every exported image and their natural distribution.
    positive = [i for i, target in enumerate(dataset.targets) if len(target['class_ids'])]
    negative = [i for i, target in enumerate(dataset.targets) if not len(target['class_ids'])]
    chosen = positive[:max(1, maximum // 2)]
    chosen += negative[:maximum - len(chosen)]
    remaining = set(chosen)
    chosen += [i for i in range(len(dataset)) if i not in remaining][:maximum - len(chosen)]
    return Subset(dataset, sorted(chosen))


def make_loader(config, split, device, maximum=0):
    dataset = ArmorPoseDataset(config['data'], split=split, input_size=tuple(config['input_size']),
                               augment=(split == 'train'), brightness=config['brightness'])
    dataset = limited_dataset(dataset, maximum)
    generator = torch.Generator().manual_seed(config['seed'])
    return DataLoader(dataset, batch_size=config['batch'], shuffle=(split == 'train'),
                      num_workers=config['workers'], collate_fn=armor_collate,
                      pin_memory=device.type == 'cuda', generator=generator,
                      persistent_workers=config['workers'] > 0)


def move_targets(targets, device):
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def loss_record(stats, loss):
    result = {key: float(value) for key, value in stats.items() if key.startswith('loss_')}
    result['loss'] = float(loss.detach())
    return result


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    model.eval()
    metrics = DetectionMetrics()
    totals, images = {}, 0
    for batch, targets, _metadata in loader:
        batch = batch.to(device, non_blocking=True)
        gpu_targets = move_targets(targets, device)
        raw = model(batch, return_raw=True)
        loss, stats = criterion(raw, gpu_targets, tuple(config['input_size']))
        for key, value in loss_record(stats, loss).items():
            totals[key] = totals.get(key, 0.) + value * len(batch)
        output = torch.cat((raw['color_logits'].sigmoid(), raw['class_logits'].sigmoid(),
                            raw['keypoints'].flatten(2)), dim=2).transpose(1, 2).cpu()
        for prediction, target in zip(output, targets):
            metrics.update(decode_predictions(prediction, config['val_conf'], config['nms_iou'], config['max_det']), target)
        images += len(batch)
    return {**metrics.compute(), **{key: value / max(images, 1) for key, value in totals.items()}}


def atomic_checkpoint(checkpoint, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run_training(config, resume=None):
    torch.set_num_threads(config['cpu_threads'])
    cv2.setNumThreads(1)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    requested = str(config['device'])
    if requested in ('cpu',):
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu for CPU training')
        device = torch.device(f'cuda:{requested}' if requested.isdigit() else ('cuda:0' if requested == 'cuda' else requested))
    if config['input_size'] != [480, 640]:
        raise ValueError('This deployment uses fixed input_size [480,640]')
    output = Path(config['output']).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'Run directory is not empty: {output}; choose --output or use --resume')
    output.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(config, 'train', device, config['max_train_images'])
    val_loader = make_loader(config, 'val', device, config['max_val_images'])
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError('Both train and val must contain images')
    print(json.dumps({'device': str(device), 'model': config.get('model', DEFAULT_MODEL), 'train_images': len(train_loader.dataset),
                      'val_images': len(val_loader.dataset), 'head': '4 colors + 9 classes + 8 XY'}, ensure_ascii=False), flush=True)
    if resume:
        model, previous = load_model(resume, device)
        if 'optimizer_state_dict' not in previous:
            raise ValueError('--resume requires last.pt with optimizer state; best.pt is for inference/export')
        requested_model = config.get('model', DEFAULT_MODEL)
        if MODEL_ALIASES.get(requested_model, requested_model) != model.architecture_config()['name']:
            raise ValueError('The selected model differs from the resume checkpoint architecture; start a new experiment')
    else:
        model = build_model(config.get('model', 'yolov8n-armor-pose'),
                            backbone_weights=config['backbone_weights'],
                            pretrained=not config['from_scratch']).to(device)
        previous = {}
    criterion = ArmorPoseLoss(**config['loss'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if 'optimizer_state_dict' in previous:
        optimizer.load_state_dict(previous['optimizer_state_dict'])
    start = int(previous.get('epoch', -1)) + 1
    best = float(previous.get('best_fitness', -math.inf))
    best_epoch = int(previous.get('best_epoch', -1))
    scaler = torch.amp.GradScaler('cuda', enabled=bool(config['amp']) and device.type == 'cuda')
    if scaler.is_enabled() and previous.get('scaler_state_dict'):
        scaler.load_state_dict(previous['scaler_state_dict'])
    if 'rng_state' in previous:
        rng = previous['rng_state']
        random.setstate(rng['python'])
        torch.set_rng_state(rng['torch'].cpu())
        numpy_rng = rng['numpy']
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32),
                             numpy_rng[2], numpy_rng[3], numpy_rng[4]))
        train_loader.generator.set_state(rng['loader'].cpu())
        if device.type == 'cuda' and rng.get('cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    if start >= config['epochs']:
        raise ValueError(f'Checkpoint already reached epoch {start}; increase --epochs')
    with open(config['data'], encoding='utf-8') as stream:
        data_config = yaml.safe_load(stream)
    with open(output / 'args.yaml', 'w', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print('epoch train_loss val_loss mAP50 mAP50_95 seconds', flush=True)
    for epoch in range(start, config['epochs']):
        begun = time.monotonic()
        model.train()
        # Linear warmup followed by cosine decay; values are based on absolute
        # epoch/step so resumed runs retain the schedule.
        factor = config['lrf'] + (1 - config['lrf']) * (1 + math.cos(math.pi * epoch / config['epochs'])) / 2
        totals, count = {}, 0
        for step, (batch, targets, _metadata) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + step + 1
            warmup = min(1., global_step / max(1, config['warmup_epochs'] * len(train_loader)))
            for group in optimizer.param_groups:
                group['lr'] = config['lr'] * factor * warmup
            batch = batch.to(device, non_blocking=True)
            targets = move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                raw = model(batch, return_raw=True)
                loss, stats = criterion(raw, targets, tuple(config['input_size']))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch + 1}, step {step + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            for key, value in loss_record(stats, loss).items():
                totals[key] = totals.get(key, 0.) + value * len(batch)
            count += len(batch)
            if (step + 1) % config['log_interval'] == 0:
                print(f'epoch {epoch + 1}/{config["epochs"]} batch {step + 1}/{len(train_loader)} loss={float(loss.detach()):.4f}', flush=True)
        result = validate(model, val_loader, criterion, device, config)
        if not math.isfinite(result['loss']):
            raise FloatingPointError(f'Non-finite validation loss at epoch {epoch + 1}')
        # Prefer AP, using validation loss as a tie breaker during cold start.
        fitness = result['mAP50_95'] - min(result['loss'], 1e6) * 1e-8
        improved = fitness > best
        if improved:
            best, best_epoch = fitness, epoch
        checkpoint = {'format': 'armor-pose-custom-v1', 'model_state_dict': model.state_dict(),
                      'architecture': model.architecture_config(), 'input_size': list(config['input_size']),
                      'preprocess': 'letterbox', 'point_names': ['LT', 'LB', 'RB', 'RT'],
                      'names': data_config.get('names', CLASS_NAMES), 'color_names': data_config.get('color_names', COLOR_NAMES),
                      'data_config': data_config, 'config': config, 'epoch': epoch,
                      'best_fitness': best, 'best_epoch': best_epoch, 'metrics': result}
        if improved:
            atomic_checkpoint(checkpoint, output / 'best.pt')
        numpy_rng = np.random.get_state()
        rng_state = {'python': random.getstate(), 'torch': torch.get_rng_state(),
                     'numpy': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                     'loader': train_loader.generator.get_state(),
                     'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        atomic_checkpoint({**checkpoint, 'optimizer_state_dict': optimizer.state_dict(),
                           'scaler_state_dict': scaler.state_dict(), 'rng_state': rng_state}, output / 'last.pt')
        row = {'epoch': epoch + 1, **{f'train_{k}': v / count for k, v in totals.items()},
               **{f'val_{k}': v for k, v in result.items() if isinstance(v, (float, int)) or v is None},
               'lr': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic() - begun}
        csv_path = output / 'results.csv'
        new_file = not csv_path.exists()
        with csv_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'{epoch + 1} {row["train_loss"]:.4f} {result["loss"]:.4f} {result["mAP50"]:.4f} {result["mAP50_95"]:.4f} {row["seconds"]:.1f}', flush=True)
        if config['patience'] > 0 and epoch - best_epoch >= config['patience']:
            print(f'Early stop: no improvement for {config["patience"]} epochs', flush=True)
            break
    print(f'Saved {output / "best.pt"} and {output / "last.pt"}', flush=True)
    return output


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help=f'YAML overrides (default: {DEFAULT_CONFIG})')
    for key in ('model', 'data', 'output', 'device', 'backbone-weights'):
        parser.add_argument('--' + key)
    for key in ('epochs', 'batch', 'workers', 'cpu-threads', 'max-train-images', 'max-val-images', 'patience'):
        parser.add_argument('--' + key, type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--from-scratch', action='store_true', default=None)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--validate', type=Path, metavar='CHECKPOINT', help='Evaluate only')
    arguments = parser.parse_args(argv)
    with DEFAULT_CONFIG.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    checkpoint_path = arguments.resume or arguments.validate
    if checkpoint_path:
        previous = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        config.update(previous.get('config', {}))
        if arguments.resume:
            if 'optimizer_state_dict' not in previous:
                parser.error('--resume requires last.pt with optimizer state')
            config['output'] = str(arguments.resume.resolve().parent)
    if arguments.config:
        with arguments.config.open(encoding='utf-8') as stream:
            config.update(yaml.safe_load(stream))
    for key, value in vars(arguments).items():
        if key not in ('config', 'resume', 'validate') and value is not None:
            config[key] = value
    for key in ('batch', 'cpu_threads', 'epochs', 'log_interval'):
        if config[key] < 1:
            parser.error(f'{key} must be positive')
    if config['workers'] < 0 or config['lr'] <= 0:
        parser.error('workers must be non-negative and lr positive')
    return arguments, config


def main(argv=None):
    args, settings = parse_config(argv)
    if args.validate:
        torch.set_num_threads(settings['cpu_threads'])
        cv2.setNumThreads(1)
        requested = str(settings['device'])
        target_device = torch.device(f'cuda:{requested}' if requested.isdigit() else requested)
        net, _ = load_model(args.validate, target_device)
        validation = validate(net, make_loader(settings, 'val', target_device, settings['max_val_images']),
                              ArmorPoseLoss(**settings['loss']), target_device, settings)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
    else:
        run_training(settings, args.resume)


if __name__ == '__main__':
    main()
