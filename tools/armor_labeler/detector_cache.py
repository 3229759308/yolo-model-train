"""Keep one detector per window so separate inference jobs can reuse its session."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Lock
from typing import Any, Callable


class DetectorCache:
    """Thread-safe construction cache; callers serialize use of detector.predict().

    Model identity includes the resolved path and file size/mtime. A complete
    configuration snapshot and factory identity also participate in the key.
    A missing file is passed to the factory, which owns model validation.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._key: tuple[Any, ...] | None = None
        self._detector: Any = None
        # Retain the callable while its id is in the key, preventing id reuse.
        self._factory: Callable[..., Any] | None = None

    def get(self, model_path: str | Path, config: dict[str, Any], *,
            factory: Callable[..., Any]) -> Any:
        with self._lock:
            path = Path(model_path).expanduser().resolve(strict=False)
            try:
                status = path.stat()
                fingerprint = (status.st_mtime_ns, status.st_size)
            except (FileNotFoundError, NotADirectoryError):
                fingerprint = None
            snapshot = copy.deepcopy(config)
            serialized = json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False)
            key = (str(path), fingerprint, serialized, id(factory))
            if key == self._key:
                return self._detector

            # Release the previous session before building another GPU model.
            # A failed factory leaves no cached detector or successful key.
            self._key = None
            self._detector = None
            self._factory = None
            detector = factory(path, snapshot)
            self._detector = detector
            self._factory = factory
            self._key = key
            return detector

    def clear(self) -> None:
        """Drop the cached instance; any outstanding caller still owns its reference."""
        with self._lock:
            self._key = None
            self._detector = None
            self._factory = None
