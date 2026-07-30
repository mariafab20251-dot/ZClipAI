import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from .models import AppConfig, ClipStyle


class WhisperConfig(BaseModel):
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 400
    })


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    n_mfcc: int = 13
    hop_length: int = 512
    n_fft: int = 2048
    vad_aggressiveness: int = 3
    min_speech_duration: float = 0.5
    max_silence_duration: float = 2.0


class VideoConfig(BaseModel):
    face_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "model": "buffalo_l",
        "det_size": [640, 640],
        "threshold": 0.5
    })
    emotion_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "model": "resnet50"
    })
    motion_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "history": 500,
        "var_threshold": 16
    })
    scene_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "threshold": 0.3,
        "min_scene_length": 15
    })


class ScoringConfig(BaseModel):
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "hook_strength": 0.25,
        "emotion_level": 0.20,
        "retention_potential": 0.20,
        "speech_energy": 0.15,
        "visual_activity": 0.10,
        "uniqueness": 0.10
    })
    thresholds: Dict[str, Any] = Field(default_factory=lambda: {
        "min_viral_score": 40,
        "min_clip_duration": 15,
        "max_clip_duration": 90,
        "overlap_threshold": 0.1,
        "min_gap_between_clips": 5.0
    })


class ShortsConfig(BaseModel):
    target_resolution: list = [1080, 1920]
    aspect_ratio: str = "9:16"
    face_tracking: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "smoothing": 0.3,
        "zoom_factor": 1.2
    })
    auto_reframe: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "padding": 0.15
    })
    auto_zoom: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "trigger_on": ["emotion_peak", "keyword", "gesture"],
        "zoom_range": [1.0, 1.5]
    })


class SubtitleConfig(BaseModel):
    enabled: bool = True
    style: str = "boxed_tiktok"  # preset ID from the new caption engine
    font_family: str = "Arial"
    font_size: int = 60
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    highlight_color: str = "#FFD700"
    position: str = "bottom"
    margin_bottom: int = 150
    max_chars_per_line: int = 32
    word_level_highlight: bool = True
    emoji_enabled: bool = True
    styles: Dict[str, Any] = Field(default_factory=lambda: {
        "tiktok": {
            "font": "Montserrat",
            "size": 65,
            "color": "#FFFFFF",
            "highlight": "#FF2D55",
            "animation": "pop"
        },
        "youtube": {
            "font": "Roboto",
            "size": 58,
            "color": "#FFFFFF",
            "highlight": "#FF0000",
            "animation": "fade"
        },
        "instagram": {
            "font": "San Francisco",
            "size": 60,
            "color": "#FFFFFF",
            "highlight": "#E1306C",
            "animation": "slide"
        }
    })


class SelectionConfig(BaseModel):
    max_clips: int = 50
    diversity_factor: float = 0.3
    prefer_complete_sentences: bool = True
    expand_boundaries: bool = True
    boundary_expansion: float = 2.0
    min_score_fallback: bool = True


class CandidatesConfig(BaseModel):
    enabled: bool = True
    min_duration: float = 15.0
    max_duration: float = 60.0
    stride: float = 6.0
    merge_gap: float = 1.5
    tail_pad: float = 0.3
    snap_silence: float = 0.15


class LLMConfig(BaseModel):
    """Optional LLM re-ranking layer (Phase 2).

    The free heuristic scorer always runs and produces a complete result. When
    `enabled` is True AND an API key is available for the selected provider, the
    top `max_candidates` heuristic-scored clips are sent to an LLM, which scores
    each 0-100 for viral potential. The two scores are then blended.

    If `enabled` is False, no key is present, or the call fails, the pipeline
    silently keeps the heuristic scores — the layer is always optional.

    API KEYS ARE NEVER STORED HERE. They are read at runtime from the
    environment variables named below (populated from a gitignored .env), so
    settings.yaml stays safe to commit.
    """
    enabled: bool = False
    provider: str = "gemini"          # gemini | openai | anthropic
    model: str = ""                   # blank -> provider default chosen at runtime
    blend_weight: float = 0.6         # 0 = ignore LLM, 1 = trust LLM only
    max_candidates: int = 25          # cost cap: only rerank the top-N heuristic clips
    transcript_chars: int = 600       # per-clip transcript truncation sent to the LLM
    timeout_seconds: float = 60.0
    # Env var names the runtime reads the key from (the KEYS themselves live in env).
    key_env: Dict[str, str] = Field(default_factory=lambda: {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    })


class ExportConfig(BaseModel):
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 20
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    threads: int = 4
    hardware_accel: bool = True


class ProcessingConfig(BaseModel):
    max_workers: int = 4
    batch_size: int = 32
    checkpoint_interval: int = 30
    resume_enabled: bool = True


class GPUConfig(BaseModel):
    enabled: bool = True
    device_id: int = 0
    fallback_to_cpu: bool = True
    memory_fraction: float = 0.8


class AppSettings(BaseModel):
    app: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "AI Viral Clipper",
        "version": "1.0.0",
        "log_level": "INFO",
        "cache_dir": "./cache",
        "temp_dir": "./temp",
        "output_dir": "./output"
    })
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    shorts: ShortsConfig = Field(default_factory=ShortsConfig)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _config: Optional[AppSettings] = None
    _config_path: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load()

    def load(self, config_path: Optional[str] = None) -> AppSettings:
        if config_path:
            self._config_path = Path(config_path)
        elif self._config_path is None:
            self._config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        if self._config_path.exists():
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = {}

        self._config = AppSettings(**data)
        self._apply_env_overrides()
        return self._config

    def _apply_env_overrides(self):
        env_mappings = {
            "WHISPER_MODEL": ("whisper", "model"),
            "WHISPER_DEVICE": ("whisper", "device"),
            "GPU_ENABLED": ("gpu", "enabled"),
            "LOG_LEVEL": ("app", "log_level"),
            "CACHE_DIR": ("app", "cache_dir"),
            "OUTPUT_DIR": ("app", "output_dir"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self._config, section)
                if hasattr(section_obj, key):
                    current_value = getattr(section_obj, key)
                    if isinstance(current_value, bool):
                        value = value.lower() in ("true", "1", "yes")
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    setattr(section_obj, key, value)

    def get(self) -> AppSettings:
        if self._config is None:
            self.load()
        return self._config

    def get_section(self, section: str) -> Any:
        section_obj = self.get()
        for part in section.split("."):
            section_obj = getattr(section_obj, part)
        if hasattr(section_obj, 'model_dump'):
            return section_obj.model_dump()
        return section_obj

    def update(self, section: str, key: str, value: Any):
        section_obj = getattr(self._config, section)
        setattr(section_obj, key, value)

    def save(self, config_path: Optional[str] = None):
        path = Path(config_path) if config_path else self._config_path
        data = self._config.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_app_config(self) -> AppConfig:
        cfg = self.get()
        return AppConfig(
            whisper_model=cfg.whisper.model,
            whisper_device=cfg.whisper.device,
            whisper_compute_type=cfg.whisper.compute_type,
            num_clips=cfg.selection.max_clips,
            clip_duration=int(cfg.candidates.min_duration),
            output_resolution=tuple(cfg.shorts.target_resolution),
            enable_gpu=cfg.gpu.enabled,
            subtitle_style=cfg.subtitles.style,
            face_tracking=cfg.shorts.face_tracking.get("enabled", True),
            auto_reframe=cfg.shorts.auto_reframe.get("enabled", True),
            min_viral_score=cfg.scoring.thresholds.get("min_viral_score", 40.0)
        )


config = ConfigManager()