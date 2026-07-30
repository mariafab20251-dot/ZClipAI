import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Callable
import numpy as np
from .logging import get_logger

logger = get_logger("cache")


class CacheManager:
    def __init__(self, cache_dir: Path, ttl: int = 86400 * 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._index_file = self.cache_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning("Failed to save cache index", error=str(e))

    def _make_key(self, *args, **kwargs) -> str:
        key_data = json.dumps([args, kwargs], sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _get_path(self, key: str, ext: str = ".pkl") -> Path:
        return self.cache_dir / f"{key}{ext}"

    def _is_expired(self, key: str) -> bool:
        if key not in self._index:
            return True
        entry = self._index[key]
        return time.time() - entry.get("timestamp", 0) > self.ttl

    def get(self, key: str, default: Any = None) -> Any:
        if self._is_expired(key):
            self.delete(key)
            return default

        path = self._get_path(key)
        if not path.exists():
            return default

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._index[key]["access_count"] = self._index[key].get("access_count", 0) + 1
            self._index[key]["last_access"] = time.time()
            self._save_index()
            return data
        except Exception as e:
            logger.warning("Cache read failed", key=key, error=str(e))
            self.delete(key)
            return default

    def set(self, key: str, value: Any, metadata: dict = None):
        path = self._get_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._index[key] = {
                "timestamp": time.time(),
                "access_count": 0,
                "last_access": time.time(),
                "metadata": metadata or {}
            }
            self._save_index()
        except Exception as e:
            logger.warning("Cache write failed", key=key, error=str(e))

    def delete(self, key: str):
        path = self._get_path(key)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        if key in self._index:
            del self._index[key]
            self._save_index()

    def clear(self):
        for path in self.cache_dir.glob("*.pkl"):
            try:
                path.unlink()
            except Exception:
                pass
        self._index = {}
        self._save_index()

    def cleanup(self):
        expired_keys = [k for k in self._index if self._is_expired(k)]
        for key in expired_keys:
            self.delete(key)
        logger.info("Cache cleanup completed", removed=len(expired_keys))

    def get_or_compute(self, key: str, compute_fn: Callable, *args, **kwargs) -> Any:
        value = self.get(key)
        if value is not None:
            logger.debug("Cache hit", key=key)
            return value

        logger.debug("Cache miss, computing", key=key)
        value = compute_fn(*args, **kwargs)
        self.set(key, value)
        return value

    def get_stats(self) -> dict:
        total_size = sum(p.stat().st_size for p in self.cache_dir.glob("*.pkl"))
        return {
            "entries": len(self._index),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }


class NumpyCacheManager(CacheManager):
    def _get_path(self, key: str, ext: str = ".npz") -> Path:
        return self.cache_dir / f"{key}{ext}"

    def set(self, key: str, value: Any, metadata: dict = None):
        path = self._get_path(key)
        try:
            if isinstance(value, dict):
                np.savez_compressed(path, **value)
            elif isinstance(value, np.ndarray):
                np.savez_compressed(path, data=value)
            elif isinstance(value, list) and value and hasattr(value[0], '__dataclass_fields__'):
                data = [v.__dict__ for v in value]
                np.savez_compressed(path, data=np.array(data, dtype=object))
            elif isinstance(value, list):
                np.savez_compressed(path, data=np.array(value, dtype=object))
            else:
                raise ValueError("NumpyCacheManager only supports dict or ndarray")

            self._index[key] = {
                "timestamp": time.time(),
                "access_count": 0,
                "last_access": time.time(),
                "metadata": metadata or {}
            }
            self._save_index()
        except Exception as e:
            logger.warning("Numpy cache write failed", key=key, error=str(e))

    def get(self, key: str, default: Any = None) -> Any:
        if self._is_expired(key):
            self.delete(key)
            return default

        path = self._get_path(key)
        if not path.exists():
            return default

        try:
            data = np.load(path, allow_pickle=True)
            result = {k: data[k] for k in data.files}
            self._index[key]["access_count"] = self._index[key].get("access_count", 0) + 1
            self._index[key]["last_access"] = time.time()
            self._save_index()
            return result
        except Exception as e:
            logger.warning("Numpy cache read failed", key=key, error=str(e))
            self.delete(key)
            return default