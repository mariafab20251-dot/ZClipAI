from pathlib import Path
from typing import List, Dict, Any, Optional
from core.models import TranscriptSegment, WordTimestamp
from utils.cuda_path import ensure_cuda_dll_path
from utils.logging import get_logger, JobLogger

logger = get_logger("transcriber")


class Transcriber:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        # Add CUDA DLL directories to PATH so faster-whisper can find
        # cublas64_12.dll at runtime.  Searches dub_venv and other known paths.
        ensure_cuda_dll_path()

        from faster_whisper import WhisperModel

        device = self.config.get("device", "cuda")
        compute_type = self.config.get("compute_type", "float16")
        model_name = self.config.get("model", "large-v3")

        # Map model name to local cache path under download_root.
        # Distil models live under a different HuggingFace repo
        # (Systran/faster-distil-whisper-*, not faster-whisper-*), so the
        # name has to be re-shaped before building the cache folder.
        if "/" not in model_name:
            if model_name.startswith("distil-"):
                hf_id = f"Systran/faster-distil-whisper-{model_name[len('distil-'):]}"
            else:
                hf_id = f"Systran/faster-whisper-{model_name}"
        else:
            hf_id = model_name
        cache_folder = hf_id.replace("/", "--")

        # Local cache lives under download_root in two possible layouts:
        #   A. HF-cache layout (what faster-whisper writes when it downloads):
        #        ./models/whisper/models--<repo with dashes>/snapshots/<hash>/model.bin
        #   B. Flat CTranslate2 folder with model.bin at the root:
        #        ./models/whisper/<repo with dashes>/model.bin
        # Resolve whichever is present so a downloaded model loads offline
        # without a huggingface_hub round-trip.
        hf_cache_dir = Path("./models/whisper") / f"models--{cache_folder}"
        flat_dir = Path("./models/whisper") / cache_folder

        local_model_path = None
        if (flat_dir / "model.bin").is_file():
            local_model_path = flat_dir
        elif hf_cache_dir.exists():
            _snaps = sorted(hf_cache_dir.glob("snapshots/*"))
            if _snaps and (_snaps[0] / "model.bin").is_file():
                local_model_path = _snaps[0]
            elif (hf_cache_dir / "model.bin").is_file():
                local_model_path = hf_cache_dir

        if local_model_path is not None:
            model_path = str(local_model_path)
            logger.info("Loading Whisper model from local path", path=model_path, device=device, compute_type=compute_type)
        else:
            model_path = model_name
            logger.info("Loading Whisper model", model=model_path, device=device, compute_type=compute_type)

        try:
            self.model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                download_root="./models/whisper"
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Whisper model", error=str(e))
            raise

    def transcribe(
        self,
        audio_path: Path,
        job_logger: JobLogger
    ) -> List[TranscriptSegment]:
        if not self.model:
            self._load_model()

        vad_filter = self.config.get("vad_filter", True)
        vad_parameters = self.config.get("vad_parameters", {})
        language = self.config.get("language", "en")
        beam_size = self.config.get("beam_size", 5)

        job_logger.info("Starting transcription", audio_path=str(audio_path))

        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0
        )

        job_logger.info("Transcription completed", language=info.language, duration=info.duration)

        transcript = []
        for i, segment in enumerate(segments):
            words = []
            if segment.words:
                for word in segment.words:
                    words.append(WordTimestamp(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        confidence=word.probability
                    ))

            transcript.append(TranscriptSegment(
                id=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
                confidence=segment.avg_logprob
            ))

        job_logger.info("Transcript parsed", segments=len(transcript))
        return transcript