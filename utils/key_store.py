"""Secure-ish local storage for API keys.

Keys are saved to `api_keys.json` next to this file's parent (project root).
That file is gitignored so keys never get committed. The GUI's "Save" button
writes here; the reranker reads from here at runtime.
"""
import json
from pathlib import Path
from typing import Dict, Optional

_KEYS_FILE = Path(__file__).resolve().parent.parent / "api_keys.json"

_PROVIDERS = ("gemini", "openai", "anthropic")


def _load() -> Dict[str, str]:
    if _KEYS_FILE.exists():
        try:
            data = json.loads(_KEYS_FILE.read_text("utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_key(provider: str, key: str) -> None:
    data = _load()
    data[provider] = key.strip()
    _KEYS_FILE.write_text(json.dumps(data, indent=2), "utf-8")


def get_key(provider: str) -> Optional[str]:
    key = _load().get(provider, "").strip()
    return key if key else None


def get_all_keys() -> Dict[str, Optional[str]]:
    data = _load()
    return {p: (data.get(p, "").strip() or None) for p in _PROVIDERS}
