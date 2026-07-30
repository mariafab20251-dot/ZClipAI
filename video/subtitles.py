"""Styled caption (ASS subtitle) generation — ported from ClipForge's caption engine.

Features:
  - 18+ trending caption presets (Hormozi, Beast, Karaoke, TikTok Boxed, etc.)
  - Word-level animations: highlight (creator style), word_reveal, one_word, karaoke
  - Auto-scaling font sizes to resolution
  - Background box, glow, shadow support
  - Backward-compatible with old settings.yaml platform styles

Single source of truth for both the ASS burn-in and the UI preview.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import Clip, WordTimestamp

logger = logging.getLogger("video.subtitles")

# ── Color conversion ──────────────────────────────────────────────────────────

_ALL_PRESET_IDS: set[str] = set()


def _hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    """Convert '#RRGGBB' to ASS '&HAABBGGRR'. alpha=0 = opaque, 255 = transparent."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    a = max(0, min(255, int(alpha)))
    return f"&H{a:02X}{b}{g}{r}".upper()


def _fmt_time(seconds: float) -> str:
    """Format seconds as ASS time H:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    cs_total = int(round(seconds * 100))
    cs = cs_total % 100
    s_total = cs_total // 100
    s = s_total % 60
    m = (s_total // 60) % 60
    h = s_total // 3600
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Escape characters special inside an ASS Dialogue field."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .strip()
    )


# ── Base preset template ──────────────────────────────────────────────────────

_BASE_PRESET = {
    "font_family": "Roboto",
    "bold": True,
    "font_size": 90,
    "primary_color": "#FFFFFF",
    "highlight_color": "#FFD400",
    "outline_color": "#000000",
    "outline": 5,
    "shadow": 1,
    "position": "bottom",
    "karaoke": False,
    "uppercase": True,
    "animation": "none",
    "tracking": 0,
    "underline": False,
    "strikethrough": False,
    "max_lines": 2,
    "max_chars": 22,
    "background_enabled": False,
    "background_color": "#000000",
    "trending": False,
}


def _P(label: str, **kw) -> dict:
    """Build a preset from the base template, overriding only what differs."""
    d = dict(_BASE_PRESET)
    d["label"] = label
    d.update(kw)
    return d


# ── Style presets — defined ONCE ──────────────────────────────────────────────

STYLE_PRESETS: dict[str, dict] = {
    "bold_white": _P("Bold White", highlight_color="#FFFFFF", font_size=96, max_chars=20),
    "karaoke_yellow": _P("Karaoke Yellow", karaoke=True, font_size=92, highlight_color="#FFE600"),
    "minimal": _P("Minimal", bold=False, uppercase=False, font_size=64, outline=1, shadow=2,
                  highlight_color="#FFFFFF", max_chars=28),
    "hormozi_green": _P("Hormozi Green", trending=True, font_family="Montserrat",
                        animation="highlight", highlight_color="#27E36B", font_size=94,
                        outline=6, position="center", max_lines=2, max_chars=16),
    "hormozi_yellow": _P("Hormozi Yellow", trending=True, font_family="Montserrat",
                         animation="highlight", highlight_color="#FFD400", font_size=94,
                         outline=6, position="center", max_lines=2, max_chars=16),
    "beast_red": _P("Beast Pop", trending=True, font_family="Anton", bold=False,
                    animation="highlight", highlight_color="#FF3B30", font_size=108,
                    outline=7, position="center", max_lines=2, max_chars=15),
    "raj_clean": _P("Raj Shamani Clean", trending=True, font_family="Poppins",
                    animation="highlight", uppercase=False, highlight_color="#FFC400",
                    font_size=78, outline=4, max_chars=26),
    "alex_caps": _P("Alex Bold Caps", trending=True, font_family="Montserrat",
                    animation="highlight", highlight_color="#22D3EE", font_size=92,
                    outline=6, position="center", max_chars=17),
    "one_word_punch": _P("One-Word Punch", trending=True, font_family="Anton", bold=False,
                         animation="one_word", font_size=132, outline=8, position="center"),
    "word_reveal": _P("Word Reveal", trending=True, font_family="Montserrat",
                      animation="word_reveal", highlight_color="#FFFFFF", font_size=90, outline=5),
    "bebas_clean": _P("Bebas Clean", trending=True, font_family="Bebas Neue", bold=False,
                      font_size=110, outline=4, highlight_color="#FFFFFF", tracking=2, max_chars=22),
    "comic_bangers": _P("Comic Punch", trending=True, font_family="Bangers", bold=False,
                        primary_color="#FFE600", highlight_color="#FFFFFF", font_size=104,
                        outline=6, max_chars=20),
    "slab_impact": _P("Slab Impact", trending=True, font_family="Alfa Slab One", bold=False,
                      animation="highlight", highlight_color="#FFD400", font_size=84, outline=6),
    "marker_note": _P("Marker", trending=True, font_family="Permanent Marker", bold=False,
                      uppercase=False, highlight_color="#FFD400", font_size=82, outline=5),
    "serif_elegant": _P("Serif Elegant", trending=True, font_family="DM Serif Display", bold=False,
                        uppercase=False, highlight_color="#FFD400", font_size=88, outline=2,
                        shadow=3, max_chars=30),
    "neon_pop": _P("Neon Pop", trending=True, font_family="Luckiest Guy", bold=False,
                   animation="highlight", highlight_color="#22D3EE", outline_color="#101018",
                   font_size=92, outline=6, position="center"),
    "boxed_tiktok": _P("Boxed", trending=True, font_family="Roboto", background_enabled=True,
                       background_color="#000000", highlight_color="#FFFFFF", font_size=78,
                       outline=6, shadow=0, max_chars=24),
    "oswald_news": _P("Oswald News", trending=True, font_family="Oswald",
                      animation="highlight", highlight_color="#FFD400", font_size=86, outline=4),
    "green_word": _P("Green Word", trending=True, font_family="Poppins",
                     animation="highlight", highlight_color="#27E36B", font_size=84, outline=5),
}

_ALL_PRESET_IDS = set(STYLE_PRESETS.keys())
DEFAULT_PRESET = "bold_white"

# Backward-compatible mapping for old settings.yaml style names (tiktok/youtube/instagram).
_OLD_STYLE_MAP = {
    "tiktok": "boxed_tiktok",
    "youtube": "bold_white",
    "instagram": "minimal",
    "generic": "bold_white",
}


def get_preset(preset_id: str) -> dict:
    """Return a preset dict, defaulting to bold_white for unknown ids."""
    resolved = _OLD_STYLE_MAP.get(preset_id, preset_id)
    return STYLE_PRESETS.get(resolved, STYLE_PRESETS[DEFAULT_PRESET])


def get_presets_list() -> list[dict]:
    """Return all presets as a list for UI display."""
    return [
        {"id": pid, **preset}
        for pid, preset in STYLE_PRESETS.items()
    ]


# ── Word grouping ─────────────────────────────────────────────────────────────

def _line_len(line: list[dict]) -> int:
    """Rendered character length of a line (words + single spaces between)."""
    if not line:
        return 0
    return sum(len(w["word"]) for w in line) + (len(line) - 1)


def _group_events(
    words: list[dict],
    max_chars: int,
    max_lines: int,
    max_span: float = 2.5,
) -> list[dict]:
    """Pack words into caption events of up to ``max_lines`` lines."""
    events: list[dict] = []
    cur_lines: list[list[dict]] = [[]]

    def event_start() -> float | None:
        for ln in cur_lines:
            if ln:
                return ln[0]["start"]
        return None

    def flush() -> None:
        nonlocal cur_lines
        filled = [ln for ln in cur_lines if ln]
        if filled:
            flat = [w for ln in filled for w in ln]
            events.append({"start": flat[0]["start"], "end": flat[-1]["end"], "lines": filled})
        cur_lines = [[]]

    for w in words:
        start = event_start()
        if start is not None and (w["end"] - start) > max_span:
            flush()
        cur_line = cur_lines[-1]
        tentative = _line_len(cur_line) + (1 if cur_line else 0) + len(w["word"])
        if cur_line and tentative > max_chars:
            if len(cur_lines) < max_lines:
                cur_lines.append([w])
            else:
                flush()
                cur_lines = [[w]]
        else:
            cur_line.append(w)
    flush()
    return events


def _word_hold_events(words: list[dict], max_gap: float = 0.7) -> list[dict]:
    """One event per word, held until the next word begins."""
    events: list[dict] = []
    n = len(words)
    for i, w in enumerate(words):
        if i + 1 < n:
            nxt = words[i + 1]["start"]
            end = nxt if (nxt - w["end"]) <= max_gap else w["end"] + 0.3
        else:
            end = w["end"]
        events.append({"start": w["start"], "end": max(end, w["end"]), "lines": [[w]]})
    return events


# ── ASS text builders ─────────────────────────────────────────────────────────

def _tok(word: str, uppercase: bool) -> str:
    t = _ass_escape(word)
    return t.upper() if uppercase else t


def _build_plain_text(lines: list[list[dict]], uppercase: bool) -> str:
    rendered = []
    for line in lines:
        rendered.append(" ".join(_tok(w["word"], uppercase) for w in line if w["word"]))
    return "\\N".join(r for r in rendered if r)


def _build_karaoke_text(lines: list[list[dict]], uppercase: bool) -> str:
    line_strs: list[str] = []
    for line in lines:
        parts: list[str] = []
        prev_end = line[0]["start"]
        for w in line:
            gap_cs = max(0, int(round((w["start"] - prev_end) * 100)))
            dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
            if gap_cs > 0:
                parts.append(f"{{\\k{gap_cs}}}")
            parts.append(f"{{\\k{dur_cs}}}{_tok(w['word'], uppercase)} ")
            prev_end = w["end"]
        line_strs.append("".join(parts).strip())
    return "\\N".join(line_strs)


def _build_reveal_text(ev: dict, uppercase: bool) -> str:
    """Word-by-word reveal with fade + pop-in via \\t."""
    ev_start = ev["start"]
    line_strs: list[str] = []
    for line in ev["lines"]:
        toks: list[str] = []
        for w in line:
            t = max(0, int(round((w["start"] - ev_start) * 1000)))
            toks.append(
                f"{{\\alpha&HFF&\\fscx70\\fscy70"
                f"\\t({t},{t + 130},\\alpha&H00&\\fscx100\\fscy100)}}"
                f"{_tok(w['word'], uppercase)}"
            )
        line_strs.append(" ".join(toks))
    return "\\N".join(line_strs)


def _build_active_word_text(ev: dict, cfg: dict, uppercase: bool) -> str:
    """Whole phrase visible; the word being spoken recolours to highlight.
    This is the signature "Hormozi" / creator look."""
    base = _hex_to_ass(cfg["primary_color"])
    hi = _hex_to_ass(cfg["highlight_color"])
    ev_start = ev["start"]
    line_strs: list[str] = []
    for line in ev["lines"]:
        toks: list[str] = []
        for w in line:
            t0 = max(0, int(round((w["start"] - ev_start) * 1000)))
            t1 = max(t0 + 1, int(round((w["end"] - ev_start) * 1000)))
            toks.append(
                f"{{\\1c{base}\\t({t0},{t0},\\1c{hi})\\t({t1},{t1},\\1c{base})}}"
                f"{_tok(w['word'], uppercase)}"
            )
        line_strs.append(" ".join(toks))
    return "\\N".join(line_strs)


def _build_one_word_text(word: dict, uppercase: bool) -> str:
    """Single word with quick fade + pop-in."""
    return (
        "{\\fad(60,0)\\fscx82\\fscy82\\t(0,130,\\fscx100\\fscy100)}"
        f"{_tok(word['word'], uppercase)}"
    )


# ── Position + override helpers ───────────────────────────────────────────────

def _override_inner(cfg: dict, video_w: int, video_h: int) -> str:
    """Build inline ASS tags (no braces) for position + rotation overrides."""
    tags: list[str] = []
    pos_x = cfg.get("pos_x")
    pos_y = cfg.get("pos_y")
    if pos_x is not None and pos_y is not None:
        x = int(round(pos_x / 100.0 * video_w))
        y = int(round(pos_y / 100.0 * video_h))
        tags.append(f"\\an5\\pos({x},{y})")
    rotation = cfg.get("rotation")
    if rotation:
        tags.append(f"\\frz{rotation:g}")
    return "".join(tags)


def _merge_overrides(preset: dict, overrides: dict | None) -> dict:
    """Layer user ``overrides`` onto ``preset``."""
    merged = dict(preset)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


# ── Main ASS builder ──────────────────────────────────────────────────────────

def build_ass(
    words: list[dict],
    style_preset: str,
    video_w: int,
    video_h: int,
    out_path: Path,
    clip_start: float = 0.0,
    overrides: dict | None = None,
) -> Path:
    """Build an .ass subtitle file at ``out_path`` and return it.

    Args:
        words: list of {"word", "start", "end"} with timings in the SOURCE timeline.
        style_preset: preset id from STYLE_PRESETS.
        video_w/video_h: target frame size (sets PlayResX/Y so font px map 1:1).
        out_path: where to write the .ass file.
        clip_start: subtract this from word timings so captions align to the cut.
        overrides: optional per-render tweaks (position, font, colours, etc.).
    """
    preset = get_preset(style_preset)
    cfg = _merge_overrides(preset, overrides)

    # Scale font/outline — presets tuned for 1080p, scale to actual resolution.
    scale = video_h / 1080.0
    font_scale = float(cfg.get("font_scale", 1.0) or 1.0)
    font_size = max(12, int(round(cfg["font_size"] * font_scale * scale)))

    outline_px = cfg.get("outline_width", cfg["outline"])
    outline = max(0, int(round(outline_px * scale)))

    shadow_on = cfg.get("shadow_enabled")
    if shadow_on is None:
        shadow_on = cfg["shadow"] > 0
    shadow_px = cfg.get("shadow_distance", cfg["shadow"])
    shadow = max(0, int(round(shadow_px * scale))) if shadow_on else 0
    shadow_color = cfg.get("shadow_color", "#000000")

    bg_on = bool(cfg.get("background_enabled"))
    border_style = 3 if bg_on else 1
    margin_v = int(round(video_h * 0.08))

    primary = _hex_to_ass(cfg["primary_color"])
    highlight = _hex_to_ass(cfg["highlight_color"])

    if bg_on:
        bg_alpha = int(round((100 - float(cfg.get("background_opacity", 100) or 100)) / 100 * 255))
        outline_col = _hex_to_ass(cfg.get("background_color", "#000000"), alpha=bg_alpha)
    else:
        outline_col = _hex_to_ass(cfg["outline_color"])

    sh_alpha = int(round((100 - float(cfg.get("shadow_opacity", 75) or 75)) / 100 * 255))
    back_col = _hex_to_ass(shadow_color, alpha=sh_alpha)
    bold_flag = -1 if cfg["bold"] else 0
    underline_flag = -1 if cfg.get("underline") else 0
    strike_flag = -1 if cfg.get("strikethrough") else 0
    spacing = max(0, int(round(float(cfg.get("tracking", 0) or 0) * scale)))
    alignment = {"top": 8, "center": 5, "bottom": 2}.get(cfg.get("position", "bottom"), 2)

    # For karaoke, PrimaryColour is the highlight, SecondaryColour the base colour.
    if cfg["karaoke"]:
        style_primary = highlight
        style_secondary = primary
    else:
        style_primary = primary
        style_secondary = primary

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cfg['font_family']},{font_size},{style_primary},{style_secondary},{outline_col},{back_col},{bold_flag},0,{underline_flag},{strike_flag},100,100,{spacing},0,{border_style},{outline},{shadow},{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    pos_inner = _override_inner(cfg, video_w, video_h)
    prefix = ("{" + pos_inner + "}") if pos_inner else ""
    uppercase = cfg["uppercase"]

    # Optional soft glow (blurred copy behind text).
    glow_on = bool(cfg.get("glow_enabled"))
    glow_px = max(1, int(round(float(cfg.get("glow_intensity", 10) or 10) * scale))) if glow_on else 0
    glow_col = _hex_to_ass(cfg.get("glow_color", "#7C4DFF"))
    glow_prefix = "{" + pos_inner + f"\\1c{glow_col}\\3c{glow_col}\\bord{glow_px}\\shad0\\blur{glow_px}" + "}"
    main_layer = 1 if glow_on else 0

    animation = cfg.get("animation") or "none"
    max_lines = int(cfg.get("max_lines") or 1)
    max_chars = int(cfg.get("max_chars") or 22)

    if animation == "one_word":
        events = _word_hold_events(words)
    else:
        events = _group_events(words, max_chars=max_chars, max_lines=max_lines)

    dialogue_rows: list[str] = []
    for ev in events:
        start = ev["start"] - clip_start
        end = ev["end"] - clip_start
        if end <= 0:
            continue
        start = max(0.0, start)

        if animation == "one_word":
            text = _build_one_word_text(ev["lines"][0][0], uppercase)
        elif animation == "word_reveal":
            text = _build_reveal_text(ev, uppercase)
        elif animation == "highlight":
            text = _build_active_word_text(ev, cfg, uppercase)
        elif cfg["karaoke"]:
            text = _build_karaoke_text(ev["lines"], uppercase)
        else:
            text = _build_plain_text(ev["lines"], uppercase)

        if glow_on:
            glow_text = _build_plain_text(ev["lines"], uppercase)
            dialogue_rows.append(
                f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Default,,0,0,0,,{glow_prefix}{glow_text}"
            )
        dialogue_rows.append(
            f"Dialogue: {main_layer},{_fmt_time(start)},{_fmt_time(end)},Default,,0,0,0,,{prefix}{text}"
        )

    out_path.write_text(header + "\n".join(dialogue_rows) + "\n", encoding="utf-8")
    return out_path


# ── SubtitleGenerator (backward-compatible API for the pipeline) ──────────────

class SubtitleGenerator:
    """Replacement subtitle generator with ClipForge's ASS engine.

    Initialised from the same settings.yaml block as before. The config values
    (font_family, font_size, font_color, etc.) are used as default overrides
    on top of the selected preset — preserving backward compatibility.
    """

    def __init__(self, subtitle_config: dict):
        self.config = subtitle_config or {}
        self.enabled = bool(subtitle_config.get("enabled", True))

    def generate(
        self,
        clip: Clip,
        output_path: Path,
        job_logger: Any = None,
        style_name: Optional[str] = None,
        hinglish: bool = False,
    ) -> Path:
        """Generate an .ass subtitle file for ``clip``.

        Args:
            clip: A Clip object with transcript, start_time, end_time, and
                  metadata["words"] (list of WordTimestamp).
            output_path: The output path (extension doesn't matter — .ass is used).
            job_logger: Optional logger.
            style_name: Preset ID or old style name (tiktok/youtube/instagram).
                        Falls back to the config's platform style, then DEFAULT_PRESET.
            hinglish: If True, transliterate Hindi/Devanagari to readable Hinglish.

        Returns:
            Path to the generated .ass file.
        """
        if not self.enabled:
            return output_path

        if job_logger:
            job_logger.info("Generating subtitles", clip_id=clip.id, output=str(output_path), hinglish=hinglish)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path = output_path.with_suffix(".ass")

        # Resolve preset ID.
        preset_id = style_name or self.config.get("style", DEFAULT_PRESET)
        resolved = _OLD_STYLE_MAP.get(preset_id, preset_id)
        if resolved not in STYLE_PRESETS:
            resolved = DEFAULT_PRESET

        # Extract word timestamps from clip metadata.
        raw_words: list = clip.metadata.get("words", [])
        if raw_words:
            words = []
            for w in raw_words:
                if hasattr(w, "word"):       # WordTimestamp object
                    words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
                else:                         # dict (from JSON serialization)
                    words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
        else:
            # Fallback: create a single "word" from the transcript.
            words = [
                {"word": clip.transcript, "start": clip.start_time, "end": clip.end_time}
            ]

        # Optionally transliterate Hindi/Devanagari to readable Hinglish.
        if hinglish:
            try:
                from indic_transliteration import sanscript

                def _to_hinglish(txt: str) -> str:
                    """ITRANS → lowercase + anusvara fix for readable Hinglish."""
                    res = sanscript.transliterate(
                        txt, sanscript.DEVANAGARI, sanscript.ITRANS
                    )
                    # Lowercase vowel-length markers and fix anusvara
                    res = (
                        res.replace("A", "a")
                        .replace("I", "i")
                        .replace("U", "u")
                        .replace("E", "e")
                        .replace("O", "o")
                        .replace("M", "n")
                        .replace("~N", "n")
                        .replace("~n", "n")
                    )
                    # Fix specific patterns for common Hinglish readability
                    res = res.replace(" men ", " mein ")
                    if res.startswith("men "):
                        res = "mein" + res[3:]
                    if res == "men":
                        res = "mein"
                    # Capitalize first letter
                    if res and res[0].isalpha():
                        res = res[0].upper() + res[1:]
                    return res

                for w in words:
                    w["word"] = _to_hinglish(w["word"])
            except Exception:
                pass  # silently skip if library not available

        # Build overrides from settings.yaml (backward compatibility).
        overrides = {}
        yaml_font = self.config.get("font_family")
        if yaml_font and yaml_font != "Arial":  # "Arial" is the old default, not meaningful
            overrides["font_family"] = yaml_font
        yaml_size = self.config.get("font_size")
        if yaml_size:
            overrides["font_size"] = int(yaml_size)
        yaml_color = self.config.get("font_color")
        if yaml_color:
            overrides["primary_color"] = yaml_color
        yaml_highlight = self.config.get("highlight_color")
        if yaml_highlight:
            overrides["highlight_color"] = yaml_highlight
        yaml_outline = self.config.get("outline_color")
        if yaml_outline:
            overrides["outline_color"] = yaml_outline
        yaml_outline_w = self.config.get("outline_width")
        if yaml_outline_w:
            overrides["outline_width"] = int(yaml_outline_w)
        yaml_pos = self.config.get("position")
        if yaml_pos:
            overrides["position"] = yaml_pos

        # Platform-specific style overrides.
        platform = self.config.get("style", "")
        style_block = self.config.get("styles", {}).get(platform, {})
        if style_block:
            font_fam = style_block.get("font", "")
            if font_fam:
                overrides["font_family"] = font_fam
            size_ov = style_block.get("size")
            if size_ov:
                overrides["font_size"] = int(size_ov)
            color_ov = style_block.get("color", "")
            if color_ov:
                overrides["primary_color"] = color_ov
            hi_ov = style_block.get("highlight", "")
            if hi_ov:
                overrides["highlight_color"] = hi_ov

        # Target resolution (default 1080×1920 for 9:16 shorts).
        video_w = self.config.get("target_width", 1080)
        video_h = self.config.get("target_height", 1920)

        # Build the .ass file.
        try:
            build_ass(
                words=words,
                style_preset=resolved,
                video_w=video_w,
                video_h=video_h,
                out_path=ass_path,
                clip_start=clip.start_time,
                overrides=overrides if overrides else None,
            )
        except Exception as e:
            logger.warning("ASS generation failed, falling back to basic style", error=str(e))
            _write_fallback_ass(ass_path, clip, video_w, video_h)

        if job_logger:
            job_logger.info("Subtitles generated", path=str(ass_path))
        return ass_path


def _write_fallback_ass(ass_path: Path, clip: Clip, video_w: int, video_h: int) -> None:
    """Minimal .ass fallback if the full engine fails."""
    start = _fmt_time(0.0)
    end = _fmt_time(clip.end_time - clip.start_time)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Roboto,72,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,10,10,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,{start},{end},Default,,0,0,0,,{_ass_escape(clip.transcript)}
"""
    ass_path.write_text(header, encoding="utf-8")
