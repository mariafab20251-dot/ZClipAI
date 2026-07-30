import subprocess
from pathlib import Path
import torch
from utils.logging import get_logger, JobLogger

logger = get_logger("audio_extractor")


class AudioExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.sample_rate = 16000

    def extract(
        self,
        video_path: Path,
        output_path: Path,
        job_logger: JobLogger
    ) -> Path:
        job_logger.info("Extracting audio", video=str(video_path), output=str(output_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            job_logger.info("Audio already extracted, using cached")
            return output_path

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                logger.error("Audio extraction failed", stderr=result.stderr)
                raise RuntimeError(f"Audio extraction failed: {result.stderr}")

            file_size = output_path.stat().st_size
            job_logger.info("Audio extracted", size_mb=round(file_size / (1024 * 1024), 2))
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out")
        except Exception as e:
            logger.error("Audio extraction error", error=str(e))
            raise

    def get_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip()) if result.stdout else 0.0