import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from utils.logging import get_logger, JobLogger
from config import config


def _subtitles_filter(subtitle_path) -> str:
    """Build a `subtitles=` filter string that points libass at the app's bundled
    font directory (assets/fonts). Without fontsdir, libass only searches
    system-installed fonts and silently falls back to a default face — which is
    why preset fonts like "Luckiest Guy" rendered as Arial in the final video."""
    sub_path_str = str(subtitle_path).replace("\\", "/")
    filter_path = sub_path_str.replace(":", "\\:")
    try:
        from video.fonts import FONTS_DIR
        fonts_dir = str(FONTS_DIR).replace("\\", "/").replace(":", "\\:")
        return f"subtitles='{filter_path}':fontsdir='{fonts_dir}'"
    except Exception:
        return f"subtitles='{filter_path}'"


# Cache NVENC availability so we only probe once per process.
_NVENC_CACHE: Optional[bool] = None


def _nvenc_available() -> bool:
    global _NVENC_CACHE
    if _NVENC_CACHE is not None:
        return _NVENC_CACHE
    try:
        r = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _NVENC_CACHE = "h264_nvenc" in r.stdout
    except Exception:
        _NVENC_CACHE = False
    return _NVENC_CACHE

logger = get_logger("video_processor")


class VideoProcessor:
    def __init__(self, export_config: Dict[str, Any], device: torch.device):
        self.config = export_config
        self.device = device
        self.codec = export_config.get("codec", "libx264")
        self.preset = export_config.get("preset", "medium")
        self.crf = export_config.get("crf", 20)
        self.audio_codec = export_config.get("audio_codec", "aac")
        self.audio_bitrate = export_config.get("audio_bitrate", "128k")
        self.threads = export_config.get("threads", 4)
        self.hw_accel = export_config.get("hardware_accel", True) and device.type == "cuda"
        # Auto-switch to NVENC when available — ~4x faster than libx264.
        # Torch may be CPU-only in this venv, so check ffmpeg directly.
        if export_config.get("hardware_accel", True) and _nvenc_available():
            self.codec = "h264_nvenc"
            self.preset = "p7"

    def export(
        self,
        video_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        job_logger: JobLogger,
        effects_config: Optional[Dict[str, Any]] = None,
        music_config: Optional[Dict[str, Any]] = None,
        border_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export video with optional cinematic effects + burned-in subtitles + background music.

        Args:
            video_path: Input video file.
            subtitle_path: Optional .ass subtitle file to burn in.
            output_path: Where to write the final mp4.
            job_logger: Logger.
            effects_config: Optional cinematic effects dict (see video/effects.py).
            music_config: Optional background-music dict with keys:
                music_path (str), volume (float), duck_amount (float).
            border_config: Optional dict with keys: enabled (bool), color (str), size (int).

        Returns:
            Path to the exported file.
        """
        job_logger.info("Exporting video", input=str(video_path), output=str(output_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_ffmpeg_command(video_path, subtitle_path, output_path, effects_config, music_config, border_config)

        # Log the actual ffmpeg command for debugging
        job_logger.info("FFmpeg command: " + " ".join(str(c) for c in cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # increased for effects processing
            )
            if result.returncode != 0:
                logger.error("FFmpeg failed", stderr=result.stderr[-2000:])
                raise RuntimeError(f"FFmpeg failed: {result.stderr[-2000:]}")

            job_logger.info("Export complete", output=str(output_path))
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError("Export timed out")
        except Exception as e:
            logger.error("Export error", error=str(e))
            raise

    def _build_ffmpeg_command(
        self,
        video_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        effects_config: Optional[Dict[str, Any]] = None,
        music_config: Optional[Dict[str, Any]] = None,
        border_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        from video.effects import build_cinematic_stages
        from video.music_mixer import build_mixer_filter

        cmd = ["ffmpeg", "-y"]

        has_sub = subtitle_path and subtitle_path.exists()
        has_border = bool(border_config and border_config.get("enabled", False))
        has_effects = bool(effects_config and any(
            v for k, v in effects_config.items() if k != "enabled" and v
        ))
        has_music = bool(music_config and music_config.get("music_path"))

        # Hardware acceleration — disabled when burning subtitles, effects, border, or music
        if self.hw_accel and not has_sub and not has_border and not has_effects and not has_music:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

        cmd.extend(["-i", str(video_path)])

        # ── Music input (if enabled) ──────────────────────────────────────────
        music_inputs: List[str] = []
        music_filter_str = ""
        music_audio_out = "0:a?"  # default = original audio passthrough
        if has_music:
            m_path = Path(music_config["music_path"])
            m_filter, m_out, m_inputs = build_mixer_filter(
                m_path,
                music_volume=music_config.get("volume", 0.30),
                duck_amount=music_config.get("duck_amount", 0.50),
                input_idx=1,
            )
            music_inputs = m_inputs
            music_filter_str = m_filter
            music_audio_out = m_out
            cmd.extend(music_inputs)

        # ── Build filtergraph stages ──────────────────────────────────────────
        stages: List[str] = []           # video filter segments
        video_out: str = "0:v"           # default video output label
        needs_filter_complex = has_music  # music always needs filter_complex

        if has_effects:
            vw = effects_config.get("target_width", 1080)
            vh = effects_config.get("target_height", 1920)
            fx_stages, fx_out = build_cinematic_stages(effects_config, "0:v", vw, vh)
            if fx_stages:
                needs_filter_complex = True
                stages.extend(fx_stages)
                video_out = f"[{fx_out}]"

        # Append subtitle burn as the final video stage (only if using filter_complex
        # or music forces us into filter_complex mode).
        if has_sub and (needs_filter_complex or has_effects):
            needs_filter_complex = True
            sub_filter = _subtitles_filter(subtitle_path)
            # If no video stages yet, start from raw input
            if not stages:
                stages.append(f"[0:v]{sub_filter}[drwv]")
                video_out = "[drwv]"
            else:
                # Strip any bracket from video_out for the input label
                in_label = video_out.strip("[]") if video_out.startswith("[") else video_out
                stages.append(f"[{in_label}]{sub_filter}[drwv]")
                video_out = "[drwv]"

        # Append border overlay (drawbox) as the last video stage.
        if has_border:
            needs_filter_complex = True
            bc_color = border_config.get("color", "#FFFFFF").lstrip("#")
            bc_thick = max(1, int(border_config.get("size", 4)))
            # Use the current video_out as input; create a new label
            in_label = video_out.strip("[]") if video_out.startswith("[") else video_out
            # drawbox strokes inward from the rectangle edges, so a full-frame
            # rectangle draws a visible border on all 4 sides (drawgrid clips the
            # right/bottom lines at the frame edge -> only top+left showed).
            stages.append(f"[{in_label}]drawbox=x=0:y=0:w=iw:h=ih:color=0x{bc_color}@1:t={bc_thick}[bordv]")
            video_out = "[bordv]"

        # Append audio mixing to the same filter_complex (if music is enabled).
        if music_filter_str:
            stages.append(music_filter_str)

        # ── Emit filter_complex (if needed) or simple -vf ─────────────────────
        if needs_filter_complex:
            cmd.extend(["-filter_complex", ";".join(stages)])
            cmd.extend(["-map", video_out, "-map", music_audio_out])
        else:
            # Simple path: subtitles and/or border (no music/effects).
            vf_filter = ""
            if has_sub:
                vf_filter = _subtitles_filter(subtitle_path)
            if has_border:
                bc_color = border_config.get("color", "#FFFFFF").lstrip("#")
                bc_thick = max(1, int(border_config.get("size", 4)))
                border_vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{bc_color}@1:t={bc_thick}"
                vf_filter = f"{vf_filter},{border_vf}" if vf_filter else border_vf
            if vf_filter:
                cmd.extend(["-vf", vf_filter])

        cmd.extend([
            "-c:v", self.codec,
            "-preset", self.preset,
        ])
        if self.codec == "h264_nvenc":
            cmd.extend(["-cq", str(self.crf), "-rc", "vbr", "-b:v", "0"])
        else:
            cmd.extend(["-crf", str(self.crf)])
        cmd.extend([
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-threads", str(self.threads),
            "-movflags", "+faststart",
            str(output_path),
        ])

        return cmd

    def extract_clip(
        self,
        input_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        job_logger: JobLogger
    ) -> Path:
        duration = end_time - start_time
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Clip extraction failed: {result.stderr}")

        return output_path

    def get_video_info(self, video_path: Path) -> dict:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        return {}