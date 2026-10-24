"""Unified command line; heavy dependencies load only for the chosen command."""
import importlib
import sys

COMMANDS = {
    'train': ('training.engine', 'Train, resume, or validate a custom-head model'),
    'predict': ('inference.predict', 'Predict images and write original-pixel labels'),
    'export': ('inference.export', 'Export a checkpoint to verified 21-channel ONNX'),
    'prepare': ('data.prepare', 'Prepare an exported annotation dataset'),
    'subsample': ('data.subsample', 'Keep all positives and a fraction of empty images'),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ('-h', '--help'):
        print('Usage: python src/armor.py COMMAND [OPTIONS]\n')
        for name, (_module, description) in COMMANDS.items():
            print(f'  {name:10s} {description}')
        print('  models     List implemented model adapters')
        print('\nUse COMMAND --help for command-specific options.')
        return 0
    command = arguments.pop(0)
    if command == 'models':
        if arguments:
            print('models takes no arguments', file=sys.stderr)
            return 2
        from .models import available_models
        for name in available_models():
            print(name)
        return 0
    if command not in COMMANDS:
        print(f'Unknown command: {command}. Use --help for available commands.', file=sys.stderr)
        return 2
    module = importlib.import_module(f'.{COMMANDS[command][0]}', package=__package__)
    result = module.main(arguments)
    return result if isinstance(result, int) else 0


if __name__ == '__main__':
    raise SystemExit(main())
