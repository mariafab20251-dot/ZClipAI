import sys
import logging
from pathlib import Path
from typing import Optional
import structlog
from structlog.types import EventDict, WrappedLogger


def add_app_context(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    event_dict["app"] = "ai_viral_clipper"
    return event_dict


def add_severity_level(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    level = method_name.upper()
    if level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        event_dict["level"] = level
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = False
) -> None:
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="ISO", utc=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_app_context,
        add_severity_level,
        timestamper,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(log_level)


def get_logger(name: str = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class JobLogger:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logger = get_logger("job").bind(job_id=job_id)

    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)

    def progress(self, step: str, progress: float, **kwargs):
        self.logger.info("progress", step=step, progress=progress, **kwargs)


def configure_third_party_loggers():
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("moviepy").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)