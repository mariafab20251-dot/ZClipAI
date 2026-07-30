# AI Viral Clipper

Automatically analyze long-form videos and generate viral short-form clips for TikTok, YouTube Shorts, Facebook Reels, and Instagram Reels.

## Features

- **AI-Powered Analysis**: Intelligent detection of viral moments using multi-modal AI
- **Smart Scoring**: 0-100 viral scoring system with configurable weights
- **Auto Reframing**: Automatic 9:16 vertical output with face tracking
- **Professional Subtitles**: Word-level timestamps with TikTok-style animations
- **GPU Acceleration**: CUDA support for faster processing
- **Smart Clip Selection**: Non-overlapping, diverse, high-scoring clips

## Architecture

```
ai_viral_clipper/
├── core/              # Core pipeline, config, models, job management
│   ├── pipeline.py    # Main orchestration pipeline
│   ├── config.py      # Configuration management
│   ├── models.py      # Data models
│   ├── exceptions.py  # Custom exceptions
│   └── job_manager.py # Job queue and persistence
├── ai/                # AI analysis modules
│   ├── transcriber.py      # Faster-Whisper transcription
│   ├── transcript_analyzer.py  # NLP analysis
│   ├── scorer.py           # Viral scoring algorithm
│   └── segment_selector.py # Clip selection logic
├── video/             # Video processing
│   ├── processor.py   # FFmpeg-based processing
│   ├── reframer.py    # 9:16 reframing with face tracking
│   ├── analyzer.py    # Visual analysis
│   └── subtitles.py   # Subtitle generation
├── audio/             # Audio processing
│   ├── extractor.py   # Audio extraction
│   └── analyzer.py    # Audio feature analysis
├── ui/                # Desktop application
│   └── main_window.py # PySide6 UI
├── utils/             # Utilities
│   ├── cache.py       # Caching system
│   ├── gpu.py         # GPU management
│   └── logging.py     # Structured logging
├── config/            # Configuration
│   └── settings.yaml  # Application settings
├── main.py            # Entry point
└── requirements.txt   # Dependencies
```

## Installation

### Prerequisites

1. **Python 3.10+**
2. **FFmpeg** (required for video/audio processing)
   - Download from https://ffmpeg.org/download.html
   - Add to system PATH

3. **CUDA Toolkit** (optional, for GPU acceleration)
   - CUDA 11.8+ recommended

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ai-viral-clipper

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
python main.py
```

## Usage

1. Launch the application
2. Select a long-form video file
3. Set desired number of clips and clip duration
4. Choose platform style (TikTok, YouTube, Instagram)
5. Configure subtitle and reframing options
6. Click "Start Processing"
7. Monitor progress and review generated clips

## Scoring System

| Component | Weight | Description |
|-----------|--------|-------------|
| Hook Strength | 25% | Opening impact, curiosity gap |
| Emotion Level | 20% | Emotional intensity |
| Retention Potential | 20% | Story completeness, payoff density |
| Speech Energy | 15% | Volume, pace, vocal variety |
| Visual Activity | 10% | Motion, scene changes, faces |
| Uniqueness | 10% | Novelty, pattern interrupts |

## Configuration

All settings are in `config/settings.yaml`. Key settings:

- **GPU**: Enable/disable, device selection, memory limits
- **Whisper**: Model selection (large-v3, medium), device, language
- **Scoring**: Weight adjustments per component
- **Shorts**: Resolution, face tracking, reframe settings
- **Subtitles**: Font, colors, animation style per platform

## License

MIT License