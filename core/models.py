from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import json


class JobStatus(Enum):
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    ANALYZING_TRANSCRIPT = "analyzing_transcript"
    ANALYZING_AUDIO = "analyzing_audio"
    ANALYZING_VIDEO = "analyzing_video"
    SCORING = "scoring"
    SELECTING_CLIPS = "selecting_clips"
    REFRAMING = "reframing"
    GENERATING_SUBTITLES = "generating_subtitles"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ClipStyle(Enum):
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    GENERIC = "generic"


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    confidence: float = 1.0
    speaker: Optional[str] = None


@dataclass
class AudioFeatures:
    segment_start: float
    segment_end: float
    rms_energy: float
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfccs: List[float]
    pitch: float
    tempo: float
    voice_activity: bool
    laughter_detected: bool = False
    applause_detected: bool = False
    peak_rms: float = 0.0          # loudest moment in the window (not the mean)
    energy_variance: float = 0.0   # dynamic range — flat vs. expressive delivery
    pitch_range: float = 0.0       # intonation spread within the window


@dataclass
class VideoFeatures:
    segment_start: float
    segment_end: float
    faces_detected: int
    face_boxes: List[List[float]] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    motion_intensity: float = 0.0
    scene_change: bool = False
    visual_activity_score: float = 0.0
    dominant_color: Optional[List[int]] = None


@dataclass
class ViralScore:
    hook_strength: float = 0.0
    emotion_level: float = 0.0
    retention_potential: float = 0.0
    speech_energy: float = 0.0
    visual_activity: float = 0.0
    uniqueness: float = 0.0
    total: float = 0.0
    bonuses: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)

    def calculate_total(self, weights: Dict[str, float]) -> float:
        self.total = (
            self.hook_strength * weights.get("hook_strength", 0.25) +
            self.emotion_level * weights.get("emotion_level", 0.20) +
            self.retention_potential * weights.get("retention_potential", 0.20) +
            self.speech_energy * weights.get("speech_energy", 0.15) +
            self.visual_activity * weights.get("visual_activity", 0.10) +
            self.uniqueness * weights.get("uniqueness", 0.10)
        )
        self.total += sum(self.bonuses.values())
        self.total -= sum(self.penalties.values())
        self.total = max(0.0, min(100.0, self.total))
        return self.total


@dataclass
class ScoredSegment:
    segment: TranscriptSegment
    viral_score: ViralScore
    audio_features: Optional[AudioFeatures] = None
    video_features: Optional[VideoFeatures] = None
    adjusted_start: Optional[float] = None
    adjusted_end: Optional[float] = None
    # Optional LLM re-ranking (Phase 2). None = LLM did not judge this segment.
    heuristic_score: Optional[float] = None  # snapshot of viral_score.total before blending
    llm_score: Optional[float] = None        # raw 0-100 from the LLM
    llm_reason: Optional[str] = None          # one-line "why this is viral"


@dataclass
class Clip:
    id: int
    start_time: float
    end_time: float
    duration: float
    viral_score: float
    transcript: str
    output_path: Optional[Path] = None
    subtitle_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    input_video: Path
    output_dir: Path
    num_clips: int
    clip_duration: int
    style: ClipStyle = ClipStyle.GENERIC
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    clips: List[Clip] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input_video": str(self.input_video),
            "output_dir": str(self.output_dir),
            "num_clips": self.num_clips,
            "clip_duration": self.clip_duration,
            "style": self.style.value,
            "status": self.status.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "clips": [
                {
                    "id": c.id,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "duration": c.duration,
                    "viral_score": c.viral_score,
                    "transcript": c.transcript,
                    "output_path": str(c.output_path) if c.output_path else None,
                    "subtitle_path": str(c.subtitle_path) if c.subtitle_path else None,
                    "metadata": c.metadata
                }
                for c in self.clips
            ],
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        job = cls(
            id=data["id"],
            input_video=Path(data["input_video"]),
            output_dir=Path(data["output_dir"]),
            num_clips=data["num_clips"],
            clip_duration=data["clip_duration"],
            style=ClipStyle(data["style"]),
            status=JobStatus(data["status"]),
            progress=data["progress"],
            current_step=data["current_step"],
            error=data["error"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            config=data["config"]
        )
        job.clips = [
            Clip(
                id=c["id"],
                start_time=c["start_time"],
                end_time=c["end_time"],
                duration=c["duration"],
                viral_score=c["viral_score"],
                transcript=c["transcript"],
                output_path=Path(c["output_path"]) if c["output_path"] else None,
                subtitle_path=Path(c["subtitle_path"]) if c["subtitle_path"] else None,
                metadata=c["metadata"]
            )
            for c in data["clips"]
        ]
        return job


@dataclass
class AppConfig:
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    num_clips: int = 10
    clip_duration: int = 30
    output_resolution: tuple = (1080, 1920)
    enable_gpu: bool = True
    subtitle_style: str = "tiktok"
    face_tracking: bool = True
    auto_reframe: bool = True
    min_viral_score: float = 40.0