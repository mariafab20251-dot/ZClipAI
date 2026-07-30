"""Auto-download trending + multilingual fonts from Google Fonts.
Ported from ClipForge's font system with all 20+ trending display fonts
and Urdu/Hindi/Arabic multilingual support.

Fonts are downloaded once and cached in assets/fonts/. The font family
name is extracted from the font's name table so libass can match it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import threading
from pathlib import Path
from typing import BinaryIO, Optional

logger = logging.getLogger("video.fonts")

# Root of the Google Fonts GitHub repo (raw content mirror).
_GF = "https://raw.githubusercontent.com/google/fonts/main"

# Project font directory — relative to the project root.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# Core fonts — downloaded synchronously; needed for default captions.
_CORE_FONTS = {
    "Roboto-Regular.ttf": "https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-Regular.ttf",
    "Roboto-Bold.ttf": "https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-Bold.ttf",
}

# Trending display fonts: family name (as embedded / used in the ASS Fontname)
# -> (local filename, download url).
TRENDING_FONTS: dict[str, tuple[str, str]] = {
    "Anton": ("Anton-Regular.ttf", f"{_GF}/ofl/anton/Anton-Regular.ttf"),
    "Bebas Neue": ("BebasNeue-Regular.ttf", f"{_GF}/ofl/bebasneue/BebasNeue-Regular.ttf"),
    "Archivo Black": ("ArchivoBlack-Regular.ttf", f"{_GF}/ofl/archivoblack/ArchivoBlack-Regular.ttf"),
    "Poppins": ("Poppins-Bold.ttf", f"{_GF}/ofl/poppins/Poppins-Bold.ttf"),
    "Montserrat": ("Montserrat.ttf", f"{_GF}/ofl/montserrat/Montserrat%5Bwght%5D.ttf"),
    "Oswald": ("Oswald.ttf", f"{_GF}/ofl/oswald/Oswald%5Bwght%5D.ttf"),
    "Teko": ("Teko.ttf", f"{_GF}/ofl/teko/Teko%5Bwght%5D.ttf"),
    "Changa": ("Changa.ttf", f"{_GF}/ofl/changa/Changa%5Bwght%5D.ttf"),
    "Bangers": ("Bangers-Regular.ttf", f"{_GF}/ofl/bangers/Bangers-Regular.ttf"),
    "Luckiest Guy": ("LuckiestGuy-Regular.ttf", f"{_GF}/apache/luckiestguy/LuckiestGuy-Regular.ttf"),
    "Permanent Marker": ("PermanentMarker-Regular.ttf", f"{_GF}/apache/permanentmarker/PermanentMarker-Regular.ttf"),
    "Alfa Slab One": ("AlfaSlabOne-Regular.ttf", f"{_GF}/ofl/alfaslabone/AlfaSlabOne-Regular.ttf"),
    "Russo One": ("RussoOne-Regular.ttf", f"{_GF}/ofl/russoone/RussoOne-Regular.ttf"),
    "Titan One": ("TitanOne-Regular.ttf", f"{_GF}/ofl/titanone/TitanOne-Regular.ttf"),
    "Paytone One": ("PaytoneOne-Regular.ttf", f"{_GF}/ofl/paytoneone/PaytoneOne-Regular.ttf"),
    "Lilita One": ("LilitaOne-Regular.ttf", f"{_GF}/ofl/lilitaone/LilitaOne-Regular.ttf"),
    "Passion One": ("PassionOne-Bold.ttf", f"{_GF}/ofl/passionone/PassionOne-Bold.ttf"),
    "Sigmar One": ("SigmarOne-Regular.ttf", f"{_GF}/ofl/sigmarone/SigmarOne-Regular.ttf"),
    "Bowlby One SC": ("BowlbyOneSC-Regular.ttf", f"{_GF}/ofl/bowlbyonesc/BowlbyOneSC-Regular.ttf"),
    "Concert One": ("ConcertOne-Regular.ttf", f"{_GF}/ofl/concertone/ConcertOne-Regular.ttf"),
    "Bungee": ("Bungee-Regular.ttf", f"{_GF}/ofl/bungee/Bungee-Regular.ttf"),
    "Shrikhand": ("Shrikhand-Regular.ttf", f"{_GF}/ofl/shrikhand/Shrikhand-Regular.ttf"),
    "DM Serif Display": ("DMSerifDisplay-Regular.ttf", f"{_GF}/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf"),
}

# Multilingual fonts for non-Latin captions (Urdu / Hindi / Arabic).
MULTILINGUAL_FONTS: dict[str, tuple[str, str]] = {
    "Noto Nastaliq Urdu": (
        "NotoNastaliqUrdu-Regular.ttf",
        f"{_GF}/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf",
    ),
    "Noto Naskh Arabic": (
        "NotoNaskhArabic-Regular.ttf",
        f"{_GF}/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf",
    ),
    "Noto Sans Arabic": (
        "NotoSansArabic-Regular.ttf",
        f"{_GF}/ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf",
    ),
    "Noto Sans Devanagari": (
        "NotoSansDevanagari-Regular.ttf",
        f"{_GF}/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
    ),
    "Noto Serif Devanagari": (
        "NotoSerifDevanagari-Regular.ttf",
        f"{_GF}/ofl/notoserifdevanagari/NotoSerifDevanagari%5Bwdth%2Cwght%5D.ttf",
    ),
}

# Language code -> default font family (avoids tofu boxes for non-Latin scripts).
LANG_DEFAULT_FONT: dict[str, str] = {
    "ur": "Noto Nastaliq Urdu",
    "ar": "Noto Sans Arabic",
    "fa": "Noto Naskh Arabic",
    "hi": "Noto Sans Devanagari",
    "mr": "Noto Sans Devanagari",
    "ne": "Noto Sans Devanagari",
}

_CORE_FAMILY_FILES = {"Roboto": "Roboto-Regular.ttf"}
ALLOWED_FONT_EXTS = {".ttf", ".otf"}
_USER_REGISTRY = FONTS_DIR / "user_fonts.json"


def _download(filename: str, url: str) -> bool:
    """Fetch one font into FONTS_DIR if missing. Returns True if present after."""
    target = FONTS_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return True
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        logger.info("Saved font %s (%d bytes).", filename, len(resp.content))
        return True
    except Exception as exc:
        logger.warning("Could not download font %s (%s).", filename, exc)
        return False


def ensure_core_fonts() -> None:
    """Download the core Roboto fonts (synchronous; needed for default render)."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in _CORE_FONTS.items():
        _download(filename, url)


def ensure_trending_fonts() -> None:
    """Download the trending + multilingual fonts (slow on first run -> thread)."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in TRENDING_FONTS.values():
        _download(filename, url)
    for filename, url in MULTILINGUAL_FONTS.values():
        _download(filename, url)
    logger.info("Trending + multilingual fonts ready in %s", FONTS_DIR)


def ensure_fonts() -> None:
    """Core fonts now (blocking), trending fonts in the background (non-blocking)."""
    ensure_core_fonts()
    threading.Thread(target=ensure_trending_fonts, daemon=True).start()


# Font family name extraction (for user uploads) — minimal sfnt 'name' table reader.
def _family_from_sfnt(data: bytes) -> Optional[str]:
    """Read the family name from a TTF/OTF byte string, or None on any problem."""
    try:
        offset = 0
        if data[:4] == b"ttcf":
            offset = struct.unpack(">I", data[12:16])[0]
        num_tables = struct.unpack(">H", data[offset + 4: offset + 6])[0]
        name_off = None
        rec = offset + 12
        for _ in range(num_tables):
            tag = data[rec: rec + 4]
            toff = struct.unpack(">I", data[rec + 8: rec + 12])[0]
            if tag == b"name":
                name_off = toff
                break
            rec += 16
        if name_off is None:
            return None
        count, string_off = struct.unpack(">HH", data[name_off + 2: name_off + 6])
        strings = name_off + string_off
        family_1 = None
        for i in range(count):
            r = name_off + 6 + i * 12
            platform, _enc, _lang, name_id, length, off = struct.unpack(">HHHHHH", data[r: r + 12])
            if name_id not in (1, 16):
                continue
            raw = data[strings + off: strings + off + length]
            try:
                text = (raw.decode("utf-16-be") if platform in (0, 3) else raw.decode("latin-1")).strip()
            except Exception:
                continue
            if not text:
                continue
            if name_id == 16:
                return text
            if family_1 is None:
                family_1 = text
        return family_1
    except Exception:
        return None


def _read_user_registry() -> list[dict]:
    if not _USER_REGISTRY.exists():
        return []
    try:
        data = json.loads(_USER_REGISTRY.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_user_registry(entries: list[dict]) -> None:
    try:
        _USER_REGISTRY.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write font registry: %s", exc)


def save_user_font(filename: str, fileobj: BinaryIO) -> dict:
    """Save an uploaded font into FONTS_DIR and register its family name."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_FONT_EXTS:
        raise ValueError(f"Unsupported font type '{ext}'. Upload a .ttf or .otf file.")
    data = fileobj.read()
    if not data:
        raise ValueError("The uploaded font file was empty.")
    family = _family_from_sfnt(data)
    if not family:
        raise ValueError("Could not read a font family name from that file. Is it a valid font?")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name) or f"font{ext}"
    dest = FONTS_DIR / safe
    try:
        dest.write_bytes(data)
    except OSError as exc:
        raise ValueError(f"Could not save the font: {exc}") from exc
    entries = [e for e in _read_user_registry() if e.get("file") != safe]
    entries.append({"family": family, "file": safe})
    _write_user_registry(entries)
    logger.info("Saved user font %s -> family '%s'", safe, family)
    return {"family": family, "file": safe}


def list_fonts() -> dict:
    """Return fonts split into bundled, trending, multilingual, and user-uploaded."""
    bundled = [{"family": fam, "file": fn} for fam, fn in _CORE_FAMILY_FILES.items()]
    bundled += [{"family": fam, "file": fn} for fam, (fn, _url) in TRENDING_FONTS.items()]
    multilingual = [{"family": fam, "file": fn} for fam, (fn, _url) in MULTILINGUAL_FONTS.items()]
    return {"bundled": bundled, "multilingual": multilingual, "user": _read_user_registry()}
