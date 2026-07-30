class AIClipperError(Exception):
    def __init__(self, message: str, code: str = "ERROR", details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class ConfigError(AIClipperError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)


class ModelLoadError(AIClipperError):
    def __init__(self, model_name: str, message: str, details: dict = None):
        super().__init__(f"Failed to load model '{model_name}': {message}", "MODEL_LOAD_ERROR", details)


class VideoProcessingError(AIClipperError):
    def __init__(self, message: str, video_path: str = None, details: dict = None):
        details = details or {}
        if video_path:
            details["video_path"] = video_path
        super().__init__(message, "VIDEO_PROCESSING_ERROR", details)


class AudioProcessingError(AIClipperError):
    def __init__(self, message: str, audio_path: str = None, details: dict = None):
        details = details or {}
        if audio_path:
            details["audio_path"] = audio_path
        super().__init__(message, "AUDIO_PROCESSING_ERROR", details)


class TranscriptionError(AIClipperError):
    def __init__(self, message: str, audio_path: str = None, details: dict = None):
        details = details or {}
        if audio_path:
            details["audio_path"] = audio_path
        super().__init__(message, "TRANSCRIPTION_ERROR", details)


class ClipGenerationError(AIClipperError):
    def __init__(self, message: str, clip_id: int = None, details: dict = None):
        details = details or {}
        if clip_id is not None:
            details["clip_id"] = clip_id
        super().__init__(message, "CLIP_GENERATION_ERROR", details)


class ExportError(AIClipperError):
    def __init__(self, message: str, output_path: str = None, details: dict = None):
        details = details or {}
        if output_path:
            details["output_path"] = output_path
        super().__init__(message, "EXPORT_ERROR", details)


class GPUError(AIClipperError):
    def __init__(self, message: str, device_id: int = None, details: dict = None):
        details = details or {}
        if device_id is not None:
            details["device_id"] = device_id
        super().__init__(message, "GPU_ERROR", details)


class ValidationError(AIClipperError):
    def __init__(self, message: str, field: str = None, value: any = None, details: dict = None):
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, "VALIDATION_ERROR", details)


class JobError(AIClipperError):
    def __init__(self, message: str, job_id: str = None, details: dict = None):
        details = details or {}
        if job_id:
            details["job_id"] = job_id
        super().__init__(message, "JOB_ERROR", details)


class CacheError(AIClipperError):
    def __init__(self, message: str, cache_key: str = None, details: dict = None):
        details = details or {}
        if cache_key:
            details["cache_key"] = cache_key
        super().__init__(message, "CACHE_ERROR", details)


class ModelDownloadError(AIClipperError):
    def __init__(self, model_name: str, message: str, details: dict = None):
        super().__init__(f"Failed to download model '{model_name}': {message}", "MODEL_DOWNLOAD_ERROR", details)