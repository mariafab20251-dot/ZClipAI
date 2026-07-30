#!/usr/bin/env python3
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from ui.main_window import MainWindow
from utils.logging import setup_logging, configure_third_party_loggers
from config import config


def main():
    app_dir = Path(__file__).parent
    os.chdir(app_dir)

    # Load API keys from a gitignored .env (GEMINI_API_KEY / OPENAI_API_KEY /
    # ANTHROPIC_API_KEY) so the optional LLM rerank layer can find them. Safe
    # no-op when python-dotenv isn't installed or no .env exists.
    try:
        from dotenv import load_dotenv
        load_dotenv(app_dir / ".env")
    except Exception:
        pass

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    Path("./cache").mkdir(parents=True, exist_ok=True)
    Path("./temp").mkdir(parents=True, exist_ok=True)
    Path("./output").mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./models/whisper").mkdir(parents=True, exist_ok=True)

    # Auto-download core fonts (Roboto) on first run; trending fonts in background.
    try:
        from video.fonts import ensure_fonts
        ensure_fonts()
    except Exception:
        pass

    settings = config.get()
    log_level = settings.app.get("log_level", "INFO")

    setup_logging(
        log_level=log_level,
        log_file=log_dir / "app.log",
        json_output=False
    )
    configure_third_party_loggers()

    app = QApplication(sys.argv)
    app.setApplicationName("AI Viral Clipper")
    app.setOrganizationName("AIClipper")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()