"""Armor dataset tools with lazy imports so preparation does not load Torch."""
from importlib import import_module


_EXPORTS = {
    'prepare_dataset': ('.prepare', 'prepare_dataset'),
    'subsample_dataset': ('.subsample', 'subsample_dataset'),
    'ArmorPoseDataset': ('.dataset', 'ArmorPoseDataset'),
    'armor_collate': ('.dataset', 'armor_collate'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
