# AI Viral Clipper - System Architecture

## Overview
Production-grade desktop application for automatic viral short-form clip generation from long-form videos.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI VIRAL CLIPPER APPLICATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │   INPUT      │   │  ANALYSIS    │   │  SCORING     │   │  OUTPUT      │  │
│  │  Pipeline    │──▶│  Pipeline    │──▶│  Engine      │──▶│  Pipeline    │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
│        │                   │                   │                   │         │
│        ▼                   ▼                   ▼                   ▼         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ Video Load   │   │ Transcript   │   │ Viral Score  │   │ Clip Cut     │  │
│  │ Audio Extract│   │ Analysis     │   │ (0-100)      │   │ Reframing    │  │
│  │ Preprocess   │   │ Audio Analysis│  │ Ranking      │   │ Subtitles    │  │
│  │ Validation   │   │ Video Analysis│  │ Deduplication│   │ Export       │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Module (`core/`)
- `pipeline.py` - Main orchestration pipeline
- `config.py` - Configuration management
- `models.py` - Data models and types
- `exceptions.py` - Custom exceptions
- `job_manager.py` - Job queue and persistence

### AI Module (`ai/`)
- `transcript_analyzer.py` - NLP-based transcript analysis
- `audio_analyzer.py` - Audio feature extraction
- `video_analyzer.py` - Visual analysis (faces, emotions, motion)
- `scorer.py` - Viral scoring algorithm
- `segment_selector.py` - Clip selection and boundary adjustment

### Video Module (`video/`)
- `processor.py` - Video processing (cut, reframe, resize)
- `face_tracker.py` - Face detection and tracking
- `reframer.py` - Auto-reframing to 9:16
- `exporter.py` - Video export with codecs

### Audio Module (`audio/`)
- `extractor.py` - Audio extraction from video
- `analyzer.py` - Audio feature analysis
- `voice_activity.py` - VAD and speech detection

### UI Module (`ui/`)
- `main_window.py` - Main application window
- `widgets/` - Custom UI components
- `worker.py` - Background processing threads
- `preview.py` - Clip preview component

### Utils Module (`utils/`)
- `gpu.py` - GPU detection and management
- `cache.py` - Caching system
- `logging.py` - Structured logging
- `helpers.py` - Utility functions

## Data Flow

```
Input Video
    │
    ▼
┌─────────────────┐
│ Job Created     │◀── User Parameters (clips, duration, style)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Extract   │──▶ WAV file (16kHz mono)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transcription   │──▶ Whisper segments with word timestamps
│ (Faster-Whisper)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Multi-Modal     │
│ Analysis        │
│  ├─ Transcript  │
│  ├─ Audio       │
│  └─ Video       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Viral Scoring   │──▶ Scored segments (0-100)
│ Algorithm       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Segment         │
│ Selection       │──▶ Top-N non-overlapping clips
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Boundary        │
│ Adjustment      │──▶ Natural start/end points
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Shorts          │
│ Optimization    │
│  ├─ 9:16 Reframe│
│  ├─ Face Track  │
│  ├─ Auto Zoom   │
│  └─ Subtitles   │
└────────┬────────┘
         │
         ▼
Output Clips
```

## AI Scoring Algorithm

### Viral Score Components (Total: 100)

| Component | Weight | Description |
|-----------|--------|-------------|
| Hook Strength | 25% | Opening impact, curiosity gap, question hooks |
| Emotion Level | 20% | Emotional intensity (excitement, surprise, anger, joy) |
| Retention Potential | 20% | Story completeness, cliffhangers, payoff density |
| Speech Energy | 15% | Volume, pace, emphasis, vocal variety |
| Visual Activity | 10% | Motion, scene changes, facial expressions |
| Uniqueness | 10% | Novelty, contrarian takes, pattern interrupts |

### Scoring Formula

```
ViralScore = Σ(ComponentScore × Weight) + Bonuses - Penalties

Bonuses:
- Laughter detected: +5
- Applause detected: +5
- Strong opinion/stance: +5
- Visual surprise: +5

Penalties:
- Low audio quality: -10
- Long pauses: -5 per occurrence
- Repetitive content: -10
- Off-topic segments: -15
```

## Technical Stack

### Core Dependencies
- **Python**: 3.10+
- **UI**: PySide6 (Qt6) - professional, native look
- **Video**: OpenCV, MoviePy, FFmpeg
- **Audio**: librosa, pydub, webrtcvad
- **AI/ML**: faster-whisper, transformers, torch, ultralytics (YOLO)
- **NLP**: spacy, textblob, vaderSentiment
- **Face**: insightface, mediapipe
- **Config**: pydantic, yaml

### GPU Acceleration
- CUDA for Whisper, YOLO, InsightFace
- Automatic fallback to CPU
- Batch processing for efficiency

### Caching Strategy
- Transcript cache (JSON)
- Audio features cache (NPZ)
- Video analysis cache (NPZ)
- Job state persistence (SQLite)

## Configuration

All settings via `config/settings.yaml` with environment variable overrides.

## Error Handling

- Structured logging (structlog)
- Graceful degradation
- Job resumption from checkpoints
- User-friendly error messages

## Testing Strategy

- Unit tests for each analyzer
- Integration tests for pipeline
- Golden master tests for scoring
- Performance benchmarks