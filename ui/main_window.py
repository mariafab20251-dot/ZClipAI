from pathlib import Path
from typing import Optional, List, Callable
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QProgressBar, QListWidget, QTextEdit, QFileDialog,
    QGroupBox, QCheckBox, QTabWidget, QSplitter,
    QMessageBox, QSlider, QScrollArea, QLineEdit,
    QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDateTime, QRectF
from PySide6.QtGui import QFont, QIcon, QPixmap, QPainter, QPainterPath, QColor, QPen, QBrush, QFontMetrics
import os
import numpy as np
from core.models import Job, JobStatus, Clip, ClipStyle
from core.job_manager import JobManager
from core.pipeline import Pipeline
from ai.llm_reranker import list_models, DEFAULT_MODELS, FALLBACK_MODEL_CHOICES, get_saved_key, save_api_key
from config import config
from utils.logging import get_logger, setup_logging, configure_third_party_loggers
from video.subtitles import STYLE_PRESETS, get_preset
from video.fonts import list_fonts

logger = get_logger("ui")


def _hex_to_qcolor(hex_str: str, default: str = "#FFFFFF") -> QColor:
    """Parse '#RRGGBB' to QColor, tolerating bad input."""
    s = (hex_str or "").strip()
    if not s:
        s = default
    c = QColor(s)
    if not c.isValid():
        c = QColor(default)
    return c


class CaptionPreviewWidget(QWidget):
    """Live WYSIWYG-ish preview of how burned-in captions will look.

    Renders a 9:16 mock video frame and paints sample caption text using the
    same preset + override values that will be sent to the ASS engine: font
    family, size (scaled to the frame like build_ass does), primary/highlight
    colour, outline, background box, uppercase and top/center/bottom position.
    It approximates the ASS output — good enough to judge look before rendering.
    """

    SAMPLE_WORDS = ["YOUR", "CAPTIONS", "WILL", "LOOK", "LIKE", "THIS"]
    HIGHLIGHT_INDEX = 3  # which sample word shows the highlight colour

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg: dict = dict(get_preset("boxed_tiktok"))
        self.setMinimumHeight(260)
        self.setMinimumWidth(200)

    def set_style(self, cfg: dict):
        self._cfg = cfg or {}
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        # ---- 9:16 mock frame, centered ----
        avail_h = self.height() - 8
        frame_h = avail_h
        frame_w = frame_h * 9 / 16
        if frame_w > self.width() - 8:
            frame_w = self.width() - 8
            frame_h = frame_w * 16 / 9
        fx = (self.width() - frame_w) / 2
        fy = (self.height() - frame_h) / 2
        frame = QRectF(fx, fy, frame_w, frame_h)

        # Video-ish backdrop (subtle vertical gradient look via two fills).
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#20242b"))
        p.drawRoundedRect(frame, 10, 10)
        p.setBrush(QColor("#2c313a"))
        p.drawRoundedRect(QRectF(fx, fy, frame_w, frame_h * 0.55), 10, 10)

        cfg = self._cfg
        uppercase = bool(cfg.get("uppercase", True))
        words = [w.upper() if uppercase else w.title() for w in self.SAMPLE_WORDS]

        # Font size scaled to the frame exactly like build_ass (scale = h/1080).
        base_size = float(cfg.get("font_size", 80))
        scale = frame_h / 1080.0
        px = max(9, int(round(base_size * scale)))
        font = QFont(cfg.get("font_family", "Arial") or "Arial")
        font.setPixelSize(px)
        font.setBold(bool(cfg.get("bold", True)))
        p.setFont(font)
        fm = QFontMetrics(font)

        # Wrap sample words into lines by max_chars (like the ASS grouper).
        max_chars = int(cfg.get("max_chars", 20) or 20)
        max_lines = max(1, int(cfg.get("max_lines", 2) or 2))
        lines: list[list[int]] = [[]]
        cur_len = 0
        for i, w in enumerate(words):
            add = len(w) + (1 if lines[-1] else 0)
            if lines[-1] and cur_len + add > max_chars and len(lines) < max_lines:
                lines.append([i]); cur_len = len(w)
            else:
                lines[-1].append(i); cur_len += add
        lines = [ln for ln in lines if ln]

        primary = _hex_to_qcolor(cfg.get("primary_color", "#FFFFFF"))
        highlight = _hex_to_qcolor(cfg.get("highlight_color", "#FFD400"), "#FFD400")
        outline_px = float(cfg.get("outline_width", cfg.get("outline", 4)) or 0)
        outline_w = max(0.0, outline_px * scale)
        outline_col = _hex_to_qcolor(cfg.get("outline_color", "#000000"), "#000000")
        bg_on = bool(cfg.get("background_enabled"))
        bg_col = _hex_to_qcolor(cfg.get("background_color", "#000000"), "#000000")

        line_h = fm.height()
        space_w = fm.horizontalAdvance(" ")
        block_h = line_h * len(lines)

        # Vertical anchor from position.
        position = cfg.get("position", "bottom")
        margin = frame_h * 0.06
        if position == "top":
            y0 = frame.top() + margin
        elif position == "center":
            y0 = frame.top() + (frame_h - block_h) / 2
        else:  # bottom
            y0 = frame.bottom() - margin - block_h

        for li, ln in enumerate(lines):
            widths = [fm.horizontalAdvance(words[i]) for i in ln]
            total_w = sum(widths) + space_w * (len(ln) - 1)
            x = frame.center().x() - total_w / 2
            y = y0 + li * line_h
            baseline = y + fm.ascent()

            if bg_on:
                pad = px * 0.12
                p.setPen(Qt.NoPen)
                p.setBrush(bg_col)
                p.drawRoundedRect(
                    QRectF(x - pad, y + (line_h - fm.height()) - pad * 0.3,
                           total_w + pad * 2, fm.height() + pad * 0.6),
                    4, 4,
                )

            cx = x
            for k, i in enumerate(ln):
                word = words[i]
                col = highlight if i == self.HIGHLIGHT_INDEX else primary
                path = QPainterPath()
                path.addText(cx, baseline, font, word)
                if outline_w > 0 and not bg_on:
                    p.setPen(QPen(outline_col, outline_w * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    p.setBrush(Qt.NoBrush)
                    p.drawPath(path)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawPath(path)
                cx += widths[k] + space_w

        p.end()


class VideoPreviewWidget(QWidget):
    """Live preview of the selected video frame with captions and effects."""

    SAMPLE_WORDS = ["YOUR", "CAPTIONS", "WILL", "LOOK", "LIKE", "THIS"]
    HIGHLIGHT_INDEX = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._cfg: dict = dict(get_preset("boxed_tiktok"))
        self._effects: dict = {}
        self._aspect = "9:16"
        self.setMinimumSize(280, 420)
        self.setStyleSheet("background: #1a1a1a; border-radius: 6px;")

    def set_video_path(self, path: Optional[Path]):
        if path and path.exists():
            try:
                import cv2
                cap = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid = max(0, total // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                cap.release()
                self._frame = frame if ret else None
            except Exception:
                self._frame = None
        else:
            self._frame = None
        self.update()

    def set_style(self, cfg: dict):
        self._cfg = cfg or {}
        self.update()

    def set_effects(self, effects: dict):
        self._effects = effects or {}
        self.update()

    def set_aspect(self, aspect: str):
        self._aspect = aspect
        self.update()

    def _parse_aspect(self) -> float:
        a = self._aspect
        if "9:16" in a or "Vertical" in a:
            return 9 / 16
        if "16:9" in a or "Landscape" in a:
            return 16 / 9
        if "4:3" in a:
            return 4 / 3
        if "1:1" in a or "Square" in a:
            return 1.0
        if "21:9" in a or "Ultrawide" in a:
            return 21 / 9
        return 9 / 16

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        ratio = self._parse_aspect()
        avail_w = self.width() - 8
        avail_h = self.height() - 8
        if ratio >= 1:
            frame_w = avail_w
            frame_h = frame_w / ratio
            if frame_h > avail_h:
                frame_h = avail_h
                frame_w = frame_h * ratio
        else:
            frame_h = avail_h
            frame_w = frame_h * ratio
            if frame_w > avail_w:
                frame_w = avail_w
                frame_h = frame_w / ratio
        fx = (self.width() - frame_w) / 2
        fy = (self.height() - frame_h) / 2
        frame = QRectF(fx, fy, frame_w, frame_h)

        if self._frame is not None:
            fh, fw = self._frame.shape[:2]
            frame_pix = QPixmap.fromImage(
                QImage(
                    cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB).data,
                    fw, fh, fw * 3, QImage.Format_RGB888
                )
            ).scaled(int(frame_w), int(frame_h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(int(fx + (frame_w - frame_pix.width()) / 2),
                         int(fy + (frame_h - frame_pix.height()) / 2), frame_pix)
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#20242b"))
            p.drawRoundedRect(frame, 10, 10)
            p.setBrush(QColor("#2c313a"))
            p.drawRoundedRect(QRectF(fx, fy, frame_w, frame_h * 0.55), 10, 10)

        self._draw_captions(p, frame)

    def _draw_captions(self, p: QPainter, frame: QRectF):
        cfg = self._cfg
        ratio = self._parse_aspect()
        uppercase = bool(cfg.get("uppercase", True))
        words = [w.upper() if uppercase else w.title() for w in self.SAMPLE_WORDS]

        base_size = float(cfg.get("font_size", 80))
        scale = frame.height() / 1080.0
        px = max(9, int(round(base_size * scale)))
        font = QFont(cfg.get("font_family", "Arial") or "Arial")
        font.setPixelSize(px)
        font.setBold(bool(cfg.get("bold", True)))
        p.setFont(font)
        fm = QFontMetrics(font)

        max_chars = int(cfg.get("max_chars", 20) or 20)
        max_lines = max(1, int(cfg.get("max_lines", 2) or 2))
        lines: list[list[int]] = [[]]
        cur_len = 0
        for i, w in enumerate(words):
            add = len(w) + (1 if lines[-1] else 0)
            if lines[-1] and cur_len + add > max_chars and len(lines) < max_lines:
                lines.append([i]); cur_len = len(w)
            else:
                lines[-1].append(i); cur_len += add
        lines = [ln for ln in lines if ln]

        primary = _hex_to_qcolor(cfg.get("primary_color", "#FFFFFF"))
        highlight = _hex_to_qcolor(cfg.get("highlight_color", "#FFD400"), "#FFD400")
        outline_px = float(cfg.get("outline_width", cfg.get("outline", 4)) or 0)
        outline_w = max(0.0, outline_px * scale)
        outline_col = _hex_to_qcolor(cfg.get("outline_color", "#000000"), "#000000")
        bg_on = bool(cfg.get("background_enabled"))
        bg_col = _hex_to_qcolor(cfg.get("background_color", "#000000"), "#000000")

        line_h = fm.height()
        space_w = fm.horizontalAdvance(" ")
        block_h = line_h * len(lines)

        position = cfg.get("position", "bottom")
        margin = frame.height() * 0.06
        if position == "top":
            y0 = frame.top() + margin
        elif position == "center":
            y0 = frame.top() + (frame.height() - block_h) / 2
        else:
            y0 = frame.bottom() - margin - block_h

        for li, ln in enumerate(lines):
            widths = [fm.horizontalAdvance(words[i]) for i in ln]
            total_w = sum(widths) + space_w * (len(ln) - 1)
            x = frame.center().x() - total_w / 2
            y = y0 + li * line_h
            baseline = y + fm.ascent()

            if bg_on:
                pad = px * 0.12
                p.setPen(Qt.NoPen)
                p.setBrush(bg_col)
                p.drawRoundedRect(
                    QRectF(x - pad, y + (line_h - fm.height()) - pad * 0.3,
                           total_w + pad * 2, fm.height() + pad * 0.6),
                    4, 4,
                )

            cx = x
            for k, i in enumerate(ln):
                word = words[i]
                col = highlight if i == self.HIGHLIGHT_INDEX else primary
                path = QPainterPath()
                path.addText(cx, baseline, font, word)
                if outline_w > 0 and not bg_on:
                    p.setPen(QPen(outline_col, outline_w * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    p.setBrush(Qt.NoBrush)
                    p.drawPath(path)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawPath(path)
                cx += widths[k] + space_w

        p.end()


class ProcessingWorker(QThread):
    progress_updated = Signal(str, float)
    clip_ready = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, pipeline: Pipeline, job: Job):
        super().__init__()
        self.pipeline = pipeline
        self.job = job

    def run(self):
        try:
            self.pipeline.run(self.job)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self.pipeline.cancel()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Viral Clipper")
        self.setMinimumSize(1200, 800)

        self.job_manager = JobManager(Path("./data/jobs.db"))
        self.current_job: Optional[Job] = None
        self.worker: Optional[ProcessingWorker] = None
        self.clip_previews: List[str] = []

        self._setup_ui()
        self._connect_signals()
        self._load_jobs_history()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        input_group = QGroupBox("Input Settings")
        input_layout = QVBoxLayout(input_group)

        video_layout = QHBoxLayout()
        self.video_label = QLabel("No video selected")
        self.select_video_btn = QPushButton("Select Video")
        video_layout.addWidget(self.video_label, 1)
        video_layout.addWidget(self.select_video_btn)

        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output: ./output")
        self.select_output_btn = QPushButton("Select Output")
        output_layout.addWidget(self.output_label, 1)
        output_layout.addWidget(self.select_output_btn)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Number of clips:"))
        self.clips_spin = QSpinBox()
        self.clips_spin.setRange(1, 50)
        self.clips_spin.setValue(10)
        params_layout.addWidget(self.clips_spin)

        params_layout.addWidget(QLabel("Clip length (s):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(15, 120)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix("s")
        params_layout.addWidget(self.duration_spin)

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Platform style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems([s.value for s in ClipStyle])
        style_layout.addWidget(self.style_combo)

        subtitle_layout = QHBoxLayout()
        self.subtitle_check = QCheckBox("Generate Subtitles")
        self.subtitle_check.setChecked(True)
        subtitle_layout.addWidget(self.subtitle_check)

        subtitle_layout.addWidget(QLabel("Aspect Ratio:"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems([
            "None (Original)",
            "9:16 Vertical (1080×1920)",
            "16:9 Landscape (1920×1080)",
            "4:3 (1440×1080)",
            "1:1 Square (1080×1080)",
            "21:9 Ultrawide (2560×1080)"
        ])
        self.aspect_combo.setCurrentText("9:16 Vertical (1080×1920)")
        subtitle_layout.addWidget(self.aspect_combo)

        input_layout.addLayout(video_layout)
        input_layout.addLayout(output_layout)
        input_layout.addLayout(params_layout)
        input_layout.addLayout(style_layout)
        input_layout.addLayout(subtitle_layout)

        # ---- AI Ranking (optional LLM rerank) -------------------------------
        ai_group = QGroupBox("AI Ranking (optional)")
        ai_layout = QVBoxLayout(ai_group)

        self.llm_check = QCheckBox("Use AI to rank clips (needs API key)")
        self.llm_check.setChecked(bool(config.get_section("llm").get("enabled", False)))
        ai_layout.addWidget(self.llm_check)

        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["gemini", "openai", "anthropic"])
        self.provider_combo.setCurrentText(config.get_section("llm").get("provider", "gemini"))
        provider_layout.addWidget(self.provider_combo, 1)
        provider_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)  # allow new models we don't know yet
        provider_layout.addWidget(self.model_combo, 1)
        self.refresh_models_btn = QPushButton("↻")
        self.refresh_models_btn.setToolTip("Fetch the live model list from the provider (uses your API key)")
        self.refresh_models_btn.setMaximumWidth(32)
        provider_layout.addWidget(self.refresh_models_btn)
        ai_layout.addLayout(provider_layout)

        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Paste your API key here")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(self.api_key_input, 1)
        self.save_key_btn = QPushButton("Save Key")
        self.save_key_btn.setMaximumWidth(70)
        key_layout.addWidget(self.save_key_btn)
        self.test_key_btn = QPushButton("Test Key")
        self.test_key_btn.setMaximumWidth(70)
        self.test_key_btn.setToolTip("Test the API key with a lightweight API call")
        key_layout.addWidget(self.test_key_btn)
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setToolTip("Show / hide API key")
        self.show_key_btn.setMaximumWidth(32)
        key_layout.addWidget(self.show_key_btn)
        ai_layout.addLayout(key_layout)

        self.llm_status_label = QLabel()
        self.llm_status_label.setWordWrap(True)
        self.llm_status_label.setStyleSheet("color: #888; font-size: 11px;")
        ai_layout.addWidget(self.llm_status_label)

        self._populate_models(live=False)
        self._update_llm_status()
        # Load saved API key into the password field
        saved = get_saved_key(self._current_provider())
        if saved:
            self.api_key_input.setText(saved)

        control_layout = QHBoxLayout()
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.cancel_btn)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.step_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.step_label)
        progress_layout.addWidget(self.progress_bar)

        left_layout.addWidget(input_group)
        left_layout.addWidget(ai_group)

        # ---- Caption Style (presets + fonts + overrides) --------------------------
        caption_group = QGroupBox("Caption Style")
        caption_layout = QVBoxLayout(caption_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for pid, ps in sorted(STYLE_PRESETS.items(), key=lambda x: x[1].get("display_name", x[0])):
            display = ps.get("display_name", pid)
            self.preset_combo.addItem(display, pid)
        current_preset = config.get_section("subtitles").get("style", "boxed_tiktok")
        idx = self.preset_combo.findData(current_preset)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        preset_row.addWidget(self.preset_combo, 1)
        caption_layout.addLayout(preset_row)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        self.font_combo.addItem("(Preset default)", "")
        fonts_info = list_fonts()
        for cat_name in ("bundled", "multilingual"):
            for f in fonts_info.get(cat_name, []):
                self.font_combo.addItem(f["family"], f["family"])
        font_row.addWidget(self.font_combo, 1)
        caption_layout.addLayout(font_row)

        cap_ov_row = QHBoxLayout()
        cap_ov_row.addWidget(QLabel("Primary:"))
        self.caption_color_input = QLineEdit()
        self.caption_color_input.setPlaceholderText("#FFFFFF")
        self.caption_color_input.setMaximumWidth(120)
        cap_ov_row.addWidget(self.caption_color_input)
        cap_ov_row.addWidget(QLabel("Highlight:"))
        self.highlight_color_input = QLineEdit()
        self.highlight_color_input.setPlaceholderText("#FFD400")
        self.highlight_color_input.setMaximumWidth(120)
        cap_ov_row.addWidget(self.highlight_color_input)
        cap_ov_row.addWidget(QLabel("Size:"))
        self.caption_size_spin = QSpinBox()
        self.caption_size_spin.setRange(0, 200)
        self.caption_size_spin.setValue(0)
        self.caption_size_spin.setSuffix(" (0=preset)")
        cap_ov_row.addWidget(self.caption_size_spin)
        caption_layout.addLayout(cap_ov_row)

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Position:"))
        self.caption_position_combo = QComboBox()
        self.caption_position_combo.addItem("Top", "top")
        self.caption_position_combo.addItem("Center", "center")
        self.caption_position_combo.addItem("Bottom", "bottom")
        # Default from the currently selected preset.
        _cur_preset = get_preset(config.get_section("subtitles").get("style", "boxed_tiktok"))
        _pidx = self.caption_position_combo.findData(_cur_preset.get("position", "bottom"))
        if _pidx >= 0:
            self.caption_position_combo.setCurrentIndex(_pidx)
        pos_row.addWidget(self.caption_position_combo, 1)
        caption_layout.addLayout(pos_row)

        self.hinglish_check = QCheckBox("Hinglish transliteration (Hindi → Romanized)")
        self.hinglish_check.setToolTip("Convert Hindi/Devanagari captions to readable Hinglish text")
        caption_layout.addWidget(self.hinglish_check)

        # Live preview of the caption look (updates as controls change).
        caption_layout.addWidget(QLabel("Preview:"))
        self.caption_preview = CaptionPreviewWidget()
        caption_layout.addWidget(self.caption_preview, 1)

        # ---- Cinematic Effects ----------------------------------------------------
        effects_group = QGroupBox("Cinematic Effects")
        effects_layout = QVBoxLayout(effects_group)

        grade_row = QHBoxLayout()
        grade_row.addWidget(QLabel("Color Grade:"))
        self.grade_combo = QComboBox()
        self.grade_combo.addItems(["None", "Warm", "Cool", "Teal/Orange", "Vintage", "Vibrant", "B&W"])
        self.grade_combo.setCurrentText("Warm")
        grade_row.addWidget(self.grade_combo, 1)
        effects_layout.addLayout(grade_row)

        fx_grid = QVBoxLayout()
        self.fx_checks = {}
        self.fx_sliders = {}
        for fx_id, fx_label in [
            ("glow", "Glow"), ("grain", "Film Grain"), ("vignette", "Vignette"),
            ("bottom_gradient", "Bottom Gradient"), ("top_gradient", "Top Gradient"),
            ("letterbox", "Letterbox Bars"), ("sharpen", "Sharpen"),
            ("chroma_shift", "Chromatic Aberration"),
        ]:
            row = QHBoxLayout()
            cb = QCheckBox(fx_label)
            self.fx_checks[fx_id] = cb
            row.addWidget(cb)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setEnabled(False)
            slider.setMaximumWidth(120)
            self.fx_sliders[fx_id] = slider
            row.addWidget(slider)
            row.addStretch()
            fx_grid.addLayout(row)
            cb.toggled.connect(lambda checked, s=slider: s.setEnabled(checked))
        effects_layout.addLayout(fx_grid)

        # Side by side: Caption Style | Cinematic Effects
        side_row = QHBoxLayout()
        side_row.addWidget(caption_group)
        side_row.addWidget(effects_group)
        left_layout.addLayout(side_row)

        # ---- Background Music ----------------------------------------------------
        music_group = QGroupBox("Background Music")
        music_layout = QVBoxLayout(music_group)

        self.music_check = QCheckBox("Enable Background Music (sidechain ducking)")
        music_layout.addWidget(self.music_check)

        music_folder_row = QHBoxLayout()
        self.music_folder_label = QLabel("No folder selected")
        self.select_music_btn = QPushButton("Select Music Folder")
        music_folder_row.addWidget(self.music_folder_label, 1)
        music_folder_row.addWidget(self.select_music_btn)
        music_layout.addLayout(music_folder_row)

        music_vol_row = QHBoxLayout()
        music_vol_row.addWidget(QLabel("Volume:"))
        self.music_volume_slider = QSlider(Qt.Horizontal)
        self.music_volume_slider.setRange(0, 100)
        self.music_volume_slider.setValue(30)
        music_vol_row.addWidget(self.music_volume_slider, 1)
        self.music_volume_label = QLabel("30%")
        music_vol_row.addWidget(self.music_volume_label)
        music_layout.addLayout(music_vol_row)

        duck_row = QHBoxLayout()
        duck_row.addWidget(QLabel("Ducking:"))
        self.music_duck_slider = QSlider(Qt.Horizontal)
        self.music_duck_slider.setRange(0, 100)
        self.music_duck_slider.setValue(50)
        duck_row.addWidget(self.music_duck_slider, 1)
        self.music_duck_label = QLabel("50%")
        duck_row.addWidget(self.music_duck_label)
        music_layout.addLayout(duck_row)

        self.music_random_check = QCheckBox("Randomize (pick random file each run)")
        self.music_random_check.setChecked(True)
        music_layout.addWidget(self.music_random_check)

        left_layout.addWidget(music_group)

        left_layout.addLayout(control_layout)
        left_layout.addWidget(progress_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        tabs = QTabWidget()
        clips_tab = QWidget()
        clips_layout = QVBoxLayout(clips_tab)
        self.clip_list = QListWidget()
        self.clip_info = QTextEdit()
        self.clip_info.setReadOnly(True)
        self.clip_info.setMaximumHeight(150)
        clips_layout.addWidget(self.clip_list)
        clips_layout.addWidget(self.clip_info)
        tabs.addTab(clips_tab, "Generated Clips")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        tabs.addTab(log_tab, "Processing Log")

        jobs_tab = QWidget()
        jobs_layout = QVBoxLayout(jobs_tab)
        self.jobs_list = QListWidget()
        jobs_layout.addWidget(self.jobs_list)
        tabs.addTab(jobs_tab, "Job History")

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(2, 2, 2, 2)
        self.video_preview = VideoPreviewWidget()
        preview_layout.addWidget(self.video_preview)
        tabs.addTab(preview_tab, "Preview")

        right_layout.addWidget(tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([420, 820])

        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self.select_video_btn.clicked.connect(self._select_video)
        self.select_output_btn.clicked.connect(self._select_output)
        self.process_btn.clicked.connect(self._start_processing)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.clip_list.currentRowChanged.connect(self._show_clip_info)
        self.jobs_list.currentRowChanged.connect(self._load_job_from_history)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.llm_check.toggled.connect(self._update_llm_status)
        self.refresh_models_btn.clicked.connect(self._refresh_models_live)
        self.save_key_btn.clicked.connect(self._save_api_key)
        self.test_key_btn.clicked.connect(self._test_api_key)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        self.select_music_btn.clicked.connect(self._select_music_folder)
        self.music_volume_slider.valueChanged.connect(
            lambda v: self.music_volume_label.setText(f"{v}%")
        )
        self.music_duck_slider.valueChanged.connect(
            lambda v: self.music_duck_label.setText(f"{v}%")
        )

        # Live caption preview — refresh whenever any caption control changes.
        self.preset_combo.currentIndexChanged.connect(self._update_caption_preview)
        self.font_combo.currentIndexChanged.connect(self._update_caption_preview)
        self.caption_color_input.textChanged.connect(self._update_caption_preview)
        self.highlight_color_input.textChanged.connect(self._update_caption_preview)
        self.caption_size_spin.valueChanged.connect(self._update_caption_preview)
        self.caption_position_combo.currentIndexChanged.connect(self._on_preset_or_position_changed)
        self._update_caption_preview()

        # Live video preview — effects and aspect ratio changes.
        self.grade_combo.currentTextChanged.connect(self._update_effects_preview)
        for fx_cb in self.fx_checks.values():
            fx_cb.toggled.connect(self._update_effects_preview)
        self.aspect_combo.currentTextChanged.connect(
            lambda txt: self.video_preview.set_aspect(txt)
        )
        self._update_effects_preview()

    def _on_preset_or_position_changed(self, *_):
        self._update_caption_preview()

    def _current_caption_style(self) -> dict:
        """Merge the selected preset with the UI overrides — the same values the
        pipeline will send to the ASS engine — for the live preview."""
        preset_id = self.preset_combo.currentData() or "boxed_tiktok"
        cfg = dict(get_preset(preset_id))
        font_family = self.font_combo.currentData() or ""
        if font_family:
            cfg["font_family"] = font_family
        primary = self.caption_color_input.text().strip()
        if primary:
            cfg["primary_color"] = primary
        highlight = self.highlight_color_input.text().strip()
        if highlight:
            cfg["highlight_color"] = highlight
        size = self.caption_size_spin.value()
        if size > 0:
            cfg["font_size"] = size
        pos = self.caption_position_combo.currentData()
        if pos:
            cfg["position"] = pos
        return cfg

    def _update_caption_preview(self, *_):
        style = self._current_caption_style()
        if hasattr(self, "caption_preview"):
            self.caption_preview.set_style(style)
        if hasattr(self, "video_preview"):
            self.video_preview.set_style(style)

    def _update_effects_preview(self, *_):
        if hasattr(self, "video_preview"):
            effects = {"grade": self.grade_combo.currentText()}
            for fx_id, cb in self.fx_checks.items():
                effects[fx_id] = cb.isChecked()
            self.video_preview.set_effects(effects)

    # ---- AI Ranking helpers --------------------------------------------------
    def _current_provider(self) -> str:
        return self.provider_combo.currentText().strip().lower()

    def _current_api_key(self) -> Optional[str]:
        # First check the GUI field
        key = self.api_key_input.text().strip()
        if key:
            return key
        # Fall back to the saved key store
        return get_saved_key(self._current_provider())

    def _populate_models(self, live: bool = False):
        provider = self._current_provider()
        key = self._current_api_key() if live else None
        models = list_models(provider, key) if live else FALLBACK_MODEL_CHOICES.get(provider, [])
        if not models:
            models = FALLBACK_MODEL_CHOICES.get(provider, [])
        current = self.model_combo.currentText().strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        # Restore prior choice, else config default, else provider default.
        cfg_model = config.get_section("llm").get("model", "")
        prefer = current or cfg_model or DEFAULT_MODELS.get(provider, "")
        if prefer:
            idx = self.model_combo.findText(prefer)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setCurrentText(prefer)
        self.model_combo.blockSignals(False)

    def _on_provider_changed(self, _text: str = ""):
        self._populate_models(live=False)
        self._update_llm_status()
        # Load saved key for the new provider into the field
        saved = get_saved_key(self._current_provider())
        self.api_key_input.setText(saved or "")

    def _refresh_models_live(self):
        if not self._current_api_key():
            self._update_llm_status()
            QMessageBox.information(
                self, "No API key",
                f"No API key found for '{self._current_provider()}'.\n\n"
                "Paste your API key above and click Save Key, then try again.",
            )
            return
        self._populate_models(live=True)
        self._update_llm_status()

    def _save_api_key(self):
        """Save the API key from the GUI field to the key store."""
        provider = self._current_provider()
        key = self.api_key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Empty key", "Paste your API key first, then click Save Key.")
            return
        path = save_api_key(provider, key)
        self._update_llm_status()
        QMessageBox.information(self, "Saved", f"API key saved for {provider}.")

    def _test_api_key(self):
        """Test the API key with a lightweight call before processing."""
        provider = self._current_provider()
        key = self.api_key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Empty key", "Enter an API key first.")
            return
        self.llm_status_label.setText(f"🔄 Testing {provider} key...")
        self.llm_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.test_key_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from ai.llm_reranker import test_api_key
            ok, msg = test_api_key(provider, key)
            if ok:
                self.llm_status_label.setText(f"✅ {msg}")
                self.llm_status_label.setStyleSheet("color: #2e7d32; font-size: 11px;")
            else:
                self.llm_status_label.setText(f"❌ {msg}")
                self.llm_status_label.setStyleSheet("color: #c62828; font-size: 11px;")
        except Exception as e:
            self.llm_status_label.setText(f"❌ Test error: {e}")
            self.llm_status_label.setStyleSheet("color: #c62828; font-size: 11px;")
        finally:
            self.test_key_btn.setEnabled(True)

    def _toggle_key_visibility(self):
        """Show or hide the plaintext API key."""
        current = self.api_key_input.echoMode()
        if current == QLineEdit.Password:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("🙈")
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("👁")

    def _update_llm_status(self, *_):
        provider = self._current_provider()
        has_key = bool(self._current_api_key())
        key_text = self.api_key_input.text().strip()
        if not self.llm_check.isChecked():
            self.llm_status_label.setText(
                "AI ranking off — using the free built-in heuristic (no key needed)."
            )
        elif has_key:
            # Warn if Gemini key doesn't look right (standard keys start with AIza)
            warning = ""
            if provider == "gemini" and key_text and not key_text.startswith("AIza"):
                warning = " ⚠ Key format unusual — Gemini keys normally start with 'AIza'. "
                warning += "If this isn't a Gemini key, the API call will fail silently."
            self.llm_status_label.setText(
                f"✓ {provider} key saved. Top clips will be re-ranked by AI.{warning}"
            )
        else:
            self.llm_status_label.setText(
                f"⚠ No {provider} API key — will fall back to the free heuristic. "
                "Type your key above and click Save Key."
            )

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv)"
        )
        if path:
            self.video_path = Path(path)
            self.video_label.setText(self.video_path.name)
            logger.info("Video selected", path=path)
            self.video_preview.set_video_path(self.video_path)

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path = Path(path)
            self.output_label.setText(f"Output: {path}")

    def _select_music_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Background Music Folder")
        if path:
            self.music_folder = Path(path)
            self.music_folder_label.setText(self.music_folder.name)
            self.music_check.setChecked(True)
            logger.info("Music folder selected", path=path)

    def _start_processing(self):
        if not hasattr(self, 'video_path') or not self.video_path.exists():
            QMessageBox.warning(self, "Error", "Please select a valid video file.")
            return

        output_dir = getattr(self, 'output_path', Path("./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save API key so the pipeline reranker can read it
        if self.llm_check.isChecked():
            key = self.api_key_input.text().strip()
            if key:
                save_api_key(self._current_provider(), key)
                # Pre-flight: test the key NOW so the user knows before processing starts
                from ai.llm_reranker import test_api_key
                ok, msg = test_api_key(self._current_provider(), key)
                if not ok:
                    retry = QMessageBox.question(
                        self, "API Key Issue",
                        f"AI ranking key test failed:\n\n{msg}\n\n"
                        "Processing will continue WITHOUT AI ranking (free heuristic only).\n\n"
                        "Fix your key and try again later.",
                        QMessageBox.Ok | QMessageBox.Cancel
                    )
                    if retry == QMessageBox.Cancel:
                        return  # user cancelled
                    # User chose to proceed without AI ranking
                    self.llm_check.setChecked(False)
            else:
                self.llm_check.setChecked(False)

        # Build caption preset + font config
        caption_preset = self.preset_combo.currentData() or ""
        font_family = self.font_combo.currentData() or ""
        caption_color = self.caption_color_input.text().strip() or ""
        highlight_color = self.highlight_color_input.text().strip() or ""
        caption_size = self.caption_size_spin.value()
        caption_position = self.caption_position_combo.currentData() or ""

        # Build effects config from GUI
        effects_cfg = {}
        grade_map = {
            "none": "", "warm": "warm", "cool": "cool",
            "teal/orange": "teal_orange", "vintage": "vintage",
            "vibrant": "vibrant", "b&w": "bw",
        }
        grade_text = self.grade_combo.currentText().lower()
        grade_key = grade_map.get(grade_text, "")
        if grade_key:
            effects_cfg["color_grade"] = grade_key
        for fx_id in self.fx_checks:
            if self.fx_checks[fx_id].isChecked():
                effects_cfg[fx_id] = True
                effects_cfg[f"{fx_id}_strength"] = self.fx_sliders[fx_id].value()

        # Build background music config
        music_cfg: Optional[dict] = None
        if self.music_check.isChecked() and hasattr(self, "music_folder") and self.music_folder:
            from video.music_mixer import pick_music
            music_file = pick_music(self.music_folder, randomize=self.music_random_check.isChecked())
            if music_file:
                music_cfg = {
                    "music_path": str(music_file),
                    "volume": self.music_volume_slider.value() / 100.0,
                    "duck_amount": self.music_duck_slider.value() / 100.0,
                }
                logger.info("Picked background music", file=str(music_file))

        job = self.job_manager.create_job(
            input_video=self.video_path,
            output_dir=output_dir,
            num_clips=self.clips_spin.value(),
            clip_duration=self.duration_spin.value(),
            config={
                "style": self.style_combo.currentText(),
                "subtitles": self.subtitle_check.isChecked(),
                "reframe": self.aspect_combo.currentText(),
                "caption_preset": caption_preset,
                "font_family": font_family,
                "caption_color": caption_color,
                "highlight_color": highlight_color,
                "caption_size": caption_size,
                "caption_position": caption_position,
                "hinglish": self.hinglish_check.isChecked(),
                "effects": effects_cfg if effects_cfg else None,
                "music": music_cfg,
                "llm": {
                    "enabled": self.llm_check.isChecked(),
                    "provider": self._current_provider(),
                    "model": self.model_combo.currentText().strip(),
                },
            }
        )

        self.current_job = job
        self._update_controls(running=True)

        pipeline = Pipeline(
            self.job_manager,
            Path("./cache"),
        )

        self.worker = ProcessingWorker(pipeline, job)
        # Route pipeline progress through Qt signal so _on_progress always
        # runs on the main thread — calling setText/setValue from a worker
        # thread silently crashes Qt (access violation, no traceback).
        self.worker.progress_updated.connect(self._on_progress)
        pipeline.progress_callback = lambda step, p: self.worker.progress_updated.emit(step, p)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self._start_time = QDateTime.currentDateTime()
        self._log("Processing started", job_id=job.id)

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self._log("Processing cancelled by user")
            self._update_controls(running=False)

    def _on_progress(self, step: str, progress: float):
        elapsed = self._start_time.secsTo(QDateTime.currentDateTime())
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        self.step_label.setText(f"{step.replace('_', ' ').title()} [{time_str}]")
        self.progress_bar.setValue(int(progress * 100))
        self._log(f"Progress: {step} ({progress*100:.0f}%) — {time_str} elapsed")

    def _on_finished(self):
        elapsed = self._start_time.secsTo(QDateTime.currentDateTime())
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        self._log("Processing completed", total_time=time_str)
        self._update_controls(running=False)
        self._refresh_clips()

        if self.current_job:
            QMessageBox.information(
                self, "Complete",
                f"Generated {len(self.current_job.clips)} clips in {time_str}!"
            )

    def _on_error(self, error_msg: str):
        self._log(f"Error: {error_msg}")
        self._update_controls(running=False)
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error_msg}")

    def _update_controls(self, running: bool):
        self.process_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.select_video_btn.setEnabled(not running)
        self.clips_spin.setEnabled(not running)
        self.duration_spin.setEnabled(not running)
        self.style_combo.setEnabled(not running)
        self.subtitle_check.setEnabled(not running)
        self.aspect_combo.setEnabled(not running)
        self.llm_check.setEnabled(not running)
        self.preset_combo.setEnabled(not running)
        self.font_combo.setEnabled(not running)
        self.caption_color_input.setEnabled(not running)
        self.highlight_color_input.setEnabled(not running)
        self.caption_size_spin.setEnabled(not running)
        self.caption_position_combo.setEnabled(not running)
        self.grade_combo.setEnabled(not running)
        for cb in self.fx_checks.values():
            cb.setEnabled(not running)
        self.music_check.setEnabled(not running)
        self.select_music_btn.setEnabled(not running)
        self.music_volume_slider.setEnabled(not running)
        self.music_duck_slider.setEnabled(not running)
        self.music_random_check.setEnabled(not running)
        self.hinglish_check.setEnabled(not running)

    def _refresh_clips(self):
        self.clip_list.clear()
        if not self.current_job:
            return

        for clip in self.current_job.clips:
            score = clip.viral_score
            stars = "★" * int(score / 10)
            item_text = f"Clip #{clip.id} | {clip.duration:.0f}s | Score: {score:.1f} {stars}"
            self.clip_list.addItem(item_text)

    def _show_clip_info(self, index: int):
        if index < 0 or not self.current_job or index >= len(self.current_job.clips):
            return

        clip = self.current_job.clips[index]
        info = f"""Clip #{clip.id}
Duration: {clip.duration:.1f}s
Viral Score: {clip.viral_score:.1f}/100
Start: {clip.start_time:.1f}s
End: {clip.end_time:.1f}s

Transcript:
{clip.transcript}

Metadata:
"""
        for key, value in clip.metadata.items():
            if key == "llm_score" and value is not None:
                info += f"  AI Rank Score: {value:.1f}/100\n"
            elif key == "llm_reason" and value:
                info += f"  AI Reason: {value}\n"
            elif key == "heuristic_score" and value is not None:
                info += f"  Heuristic score: {value:.1f}/100\n"
            else:
                info += f"  {key}: {value:.2f}\n" if isinstance(value, float) else f"  {key}: {value}\n"

        if clip.output_path:
            info += f"\nOutput: {clip.output_path}"

        self.clip_info.setText(info)

    def _load_jobs_history(self):
        jobs = self.job_manager.get_jobs(limit=20)
        self.jobs_list.clear()
        for job in jobs:
            self.jobs_list.addItem(f"{job.id} - {job.input_video.name} ({job.status.value})")

    def _load_job_from_history(self, index: int):
        if index < 0:
            return
        jobs = self.job_manager.get_jobs(limit=20)
        if index < len(jobs):
            self.current_job = jobs[index]
            self._refresh_clips()

    def _log(self, message: str, **kwargs):
        ts = QDateTime.currentDateTime().toString("HH:mm:ss")
        if kwargs:
            parts = [message]
            for key, value in kwargs.items():
                parts.append(f"  {key}: {value}")
            message = "\n".join(parts)
        self.log_text.append(f"[{ts}] {message}")
        logger.info(message)