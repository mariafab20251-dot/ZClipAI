"""Background music selection + sidechain ducking for video exports.
Ported from ClipForge's music ducking system.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

MUSIC_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac"}


def find_music_files(folder: Path) -> List[Path]:
    """Return all music files in *folder* (non-recursive), sorted."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in MUSIC_EXTENSIONS)


def pick_music(folder: Path, randomize: bool = True) -> Optional[Path]:
    """Pick one music file from the folder."""
    files = find_music_files(folder)
    if not files:
        return None
    if randomize and len(files) > 1:
        return random.choice(files)
    return files[0]


def build_mixer_filter(
    music_path: Path,
    *,
    music_volume: float = 0.30,
    duck_amount: float = 0.50,
    input_idx: int = 1,
) -> Tuple[str, str, List[str]]:
    """Build a filter_complex segment for sidechain-ducked background music.

    Args:
        music_path: Path to the background music file.
        music_volume: 0.0-1.0 volume of the music track.
        duck_amount: 0.0-1.0 ducking intensity (0 = no ducking, 1 = max).
        input_idx: ffmpeg input index of the music file (default 1).

    Returns:
        (filter_string, audio_output_label, extra_input_args)
    """
    vol = max(0.0, min(1.0, music_volume))
    duck = max(0.0, min(1.0, duck_amount))

    # Map duck_amount to sidechaincompress params.
    # ffmpeg threshold range = -60dB to 0dB (linear 0.001 to 1.0).
    #   0.0 duck → threshold=-20dB (only ducks on loud speech, ratio 1:1 = no compression)
    #   0.5 duck → threshold=-35dB (moderate ducking)
    #   1.0 duck → threshold=-50dB (even quiet speech triggers ducking, 20:1)
    threshold = max(-50.0, -20.0 - duck * 30.0)
    ratio = 1.0 + duck * 19.0
    attack = 5.0
    release = 50.0 + (1.0 - duck) * 200.0

    filter_parts = [
        f"[{input_idx}:a]volume={vol}[bgm]",
        f"[0:a][bgm]sidechaincompress="
        f"threshold={threshold:.0f}dB:"
        f"ratio={ratio:.1f}:"
        f"attack={attack:.0f}ms:"
        f"release={release:.0f}ms"
        f"[mixed_a]",
    ]

    return ";".join(filter_parts), "mixed_a", ["-i", str(music_path)]
