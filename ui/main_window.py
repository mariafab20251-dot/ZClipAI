from pathlib import Path
from typing import Optional, List, Callable
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QProgressBar, QListWidget, QTextEdit, QFileDialog,
    QGroupBox, QCheckBox, QTabWidget, QSplitter, QGridLayout,
    QMessageBox, QSlider, QScrollArea, QLineEdit,
    QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDateTime, QRectF
from PySide6.QtGui import QFont, QIcon, QPixmap, QPainter, QPainterPath, QColor, QPen, QBrush, QFontMetrics
import os
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
        self.setMinimumHeight(140)
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


class VideoPreviewWidget(QWidget):
    """Large WYSIWYG preview showing actual video frame + effects + captions."""

    SAMPLE_WORDS = ["YOUR", "CAPTIONS", "WILL", "LOOK", "LIKE", "THIS"]
    HIGHLIGHT_INDEX = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._cfg: dict = dict(get_preset("boxed_tiktok"))
        self._effects: dict = {}
        self.setMinimumSize(280, 420)
        self.setStyleSheet("background: #1a1a1a; border-radius: 6px;")

    def set_video_path(self, path: Optional[Path]):
        if path and path.exists():
            import cv2
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid = max(0, total // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            self._frame = frame if ret else None
        else:
            self._frame = None
        self.update()

    def set_style(self, cfg: dict):
        self._cfg = cfg or {}
        self.update()

    def set_effects(self, effects: dict):
        self._effects = effects or {}
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fit frame into widget maintaining 9:16
        avail_w = self.width() - 8
        avail_h = self.height() - 8
        if self._frame is not None:
            fh, fw = self._frame.shape[:2]
        else:
            fw, fh = 1080, 1920
        scale = min(avail_w / fw, avail_h / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        dx = (self.width() - dw) // 2
        dy = (self.height() - dh) // 2

        frame_rect = QRectF(dx, dy, dw, dh)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#20242b"))
        p.drawRoundedRect(frame_rect, 8, 8)

        if self._frame is not None:
            # Apply basic color grade approximation
            img = self._apply_effects(self._frame.copy())
            # Convert BGR → RGB → QImage
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h2, w2 = rgb.shape[:2]
            qimg = QImage(rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(dx, dy, pix)
        else:
            p.setBrush(QColor("#2c313a"))
            p.drawRoundedRect(QRectF(dx, dy, dw, dh * 0.55), 8, 8)

        # Overlay captions
        self._draw_captions(p, dw, dh, dx, dy)

    def _apply_effects(self, img):
        eff = self._effects
        grade = eff.get("grade", "None")
        if grade == "Warm":
            img = cv2.addWeighted(img, 1, img, 0, 15)
            img[:, :, 2] = cv2.add(img[:, :, 2], 20)  # boost R
        elif grade == "Cool":
            img[:, :, 0] = cv2.add(img[:, :, 0], 20)   # boost B
        elif grade == "B&W":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif grade == "Vibrant":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 30)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif grade == "Vintage":
            img[:, :, 0] = cv2.multiply(img[:, :, 0], 0.85)
            img[:, :, 2] = cv2.add(img[:, :, 2], 25)

        if eff.get("vignette"):
            self._apply_vignette(img)
        if eff.get("grain"):
            self._apply_grain(img)
        if eff.get("sharpen"):
            self._apply_sharpen(img)
        return img

    def _apply_vignette(self, img):
        h, w = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w * 0.3)
        kernel_y = cv2.getGaussianKernel(h, h * 0.3)
        mask = kernel_y * kernel_x.T
        mask = mask / mask.max()
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * mask).astype(np.uint8)

    def _apply_grain(self, img):
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return img

    def _apply_sharpen(self, img):
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    def _draw_captions(self, p, dw, dh, dx, dy):
        cfg = self._cfg
        uppercase = bool(cfg.get("uppercase", True))
        words = [w.upper() if uppercase else w.title() for w in self.SAMPLE_WORDS]

        base_size = float(cfg.get("font_size", 80))
        scale = dh / 1920.0
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
        outline_color = _hex_to_qcolor(cfg.get("outline_color", "#000000"))
        outline_w = max(0, int(cfg.get("outline_width", 3) * scale))
        bg_color = _hex_to_qcolor(cfg.get("background_color", "#000000"), "#000000")
        bg_alpha = min(255, max(0, int(float(cfg.get("background_alpha", 0.6)) * 255)))

        pos = cfg.get("position", "bottom")
        line_h = fm.height() + 8
        total_h = len(lines) * line_h
        margin = int(40 * scale)

        if pos == "top":
            base_y = dy + margin + line_h
        elif pos == "center":
            base_y = dy + (dh - total_h) // 2 + line_h
        else:
            base_y = dy + dh - margin - total_h + line_h

        pad = int(12 * scale)
        box_h = int(line_h * 0.85)
        round_r = int(8 * scale)

        for line_words in lines:
            text_parts = [words[i] for i in line_words]
            full_text = " ".join(text_parts)
            text_w = fm.horizontalAdvance(full_text) + pad * 2

            bx = dx + (dw - text_w) // 2
            by = base_y - box_h

            # Background box
            p.setPen(Qt.NoPen)
            bg = QColor(bg_color)
            bg.setAlpha(bg_alpha)
            p.setBrush(bg)
            p.drawRoundedRect(bx, by, text_w, box_h, round_r, round_r)

            # Word-level coloring
            cx = bx + pad
            for i, w in enumerate(line_words):
                color = highlight if w == self.SAMPLE_WORDS[self.HIGHLIGHT_INDEX] else primary
                p.setPen(color)
                p.drawText(cx, base_y, words[i])
                cx += fm.horizontalAdvance(words[i]) + fm.horizontalAdvance(" ")
            base_y += line_h
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
        input_layout.setSpacing(3)
        input_layout.setContentsMargins(6, 6, 6, 6)

        row1 = QHBoxLayout()
        self.video_label = QLabel("No video selected")
        self.select_video_btn = QPushButton("Select Video")
        self.output_label = QLabel("Output Dir")
        self.select_output_btn = QPushButton("Browse")
        row1.addWidget(self.video_label, 1)
        row1.addWidget(self.select_video_btn)
        row1.addWidget(self.output_label)
        row1.addWidget(self.select_output_btn)
        input_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Clips:"))
        self.clips_spin = QSpinBox()
        self.clips_spin.setRange(1, 50)
        self.clips_spin.setValue(10)
        self.clips_spin.setMaximumWidth(55)
        row2.addWidget(self.clips_spin)
        row2.addWidget(QLabel("Length:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(15, 120)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix("s")
        self.duration_spin.setMaximumWidth(65)
        row2.addWidget(self.duration_spin)
        row2.addWidget(QLabel("Aspect:"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems([
            "None", "9:16", "16:9", "4:3", "1:1", "21:9"
        ])
        self.aspect_combo.setCurrentText("9:16")
        self.aspect_combo.setMaximumWidth(80)
        row2.addWidget(self.aspect_combo)
        input_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems([s.value for s in ClipStyle])
        self.style_combo.setMaximumWidth(100)
        row3.addWidget(self.style_combo)
        self.subtitle_check = QCheckBox("Subtitles")
        self.subtitle_check.setChecked(True)
        row3.addWidget(self.subtitle_check)
        input_layout.addLayout(row3)

        # ---- AI Ranking (optional LLM rerank) -------------------------------
        ai_group = QGroupBox("AI Ranking")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(3)
        ai_layout.setContentsMargins(6, 6, 6, 6)

        self.llm_check = QCheckBox("Use AI to rank clips (needs API key)")
        self.llm_check.setChecked(bool(config.get_section("llm").get("enabled", False)))
        ai_layout.addWidget(self.llm_check)

        row_prov = QHBoxLayout()
        row_prov.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["gemini", "openai", "anthropic"])
        self.provider_combo.setCurrentText(config.get_section("llm").get("provider", "gemini"))
        self.provider_combo.setMaximumWidth(100)
        row_prov.addWidget(self.provider_combo)
        row_prov.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMaximumWidth(110)
        row_prov.addWidget(self.model_combo)
        self.refresh_models_btn = QPushButton("↻")
        self.refresh_models_btn.setToolTip("Fetch live model list from the provider (uses your API key)")
        self.refresh_models_btn.setMaximumWidth(28)
        row_prov.addWidget(self.refresh_models_btn)
        ai_layout.addLayout(row_prov)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Paste your API key here")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self.api_key_input)
        self.save_key_btn = QPushButton("Save")
        self.save_key_btn.setMaximumWidth(50)
        key_row.addWidget(self.save_key_btn)
        self.test_key_btn = QPushButton("Test")
        self.test_key_btn.setMaximumWidth(50)
        self.test_key_btn.setToolTip("Test the API key with a lightweight API call")
        key_row.addWidget(self.test_key_btn)
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setToolTip("Show / hide API key")
        self.show_key_btn.setMaximumWidth(28)
        key_row.addWidget(self.show_key_btn)
        ai_layout.addLayout(key_row)

        self.llm_status_label = QLabel()
        self.llm_status_label.setWordWrap(True)
        self.llm_status_label.setStyleSheet("color: #888; font-size: 11px;")
        ai_layout.addWidget(self.llm_status_label)

        self._populate_models(live=False)
        self._update_llm_status()
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

        # ---- Caption Style (presets + fonts + overrides) --------------------------
        caption_group = QGroupBox("Caption Style")
        caption_layout = QVBoxLayout(caption_group)
        caption_layout.setSpacing(3)
        caption_layout.setContentsMargins(6, 6, 6, 6)

        row_pf = QHBoxLayout()
        row_pf.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for pid, ps in sorted(STYLE_PRESETS.items(), key=lambda x: x[1].get("display_name", x[0])):
            display = ps.get("display_name", pid)
            self.preset_combo.addItem(display, pid)
        current_preset = config.get_section("subtitles").get("style", "boxed_tiktok")
        idx = self.preset_combo.findData(current_preset)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.setMaximumWidth(110)
        row_pf.addWidget(self.preset_combo)
        row_pf.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        self.font_combo.addItem("(Default)", "")
        fonts_info = list_fonts()
        for cat_name in ("bundled", "multilingual"):
            for f in fonts_info.get(cat_name, []):
                self.font_combo.addItem(f["family"], f["family"])
        self.font_combo.setMaximumWidth(100)
        row_pf.addWidget(self.font_combo)
        caption_layout.addLayout(row_pf)

        row_clr = QHBoxLayout()
        row_clr.addWidget(QLabel("Color:"))
        self.caption_color_input = QLineEdit()
        self.caption_color_input.setPlaceholderText("#FFF")
        self.caption_color_input.setMaximumWidth(60)
        row_clr.addWidget(self.caption_color_input)
        row_clr.addWidget(QLabel("Hl:"))
        self.highlight_color_input = QLineEdit()
        self.highlight_color_input.setPlaceholderText("#FFD400")
        self.highlight_color_input.setMaximumWidth(60)
        row_clr.addWidget(self.highlight_color_input)
        row_clr.addWidget(QLabel("Sz:"))
        self.caption_size_spin = QSpinBox()
        self.caption_size_spin.setRange(0, 200)
        self.caption_size_spin.setValue(0)
        self.caption_size_spin.setSuffix("")
        self.caption_size_spin.setMaximumWidth(50)
        row_clr.addWidget(self.caption_size_spin)
        row_clr.addWidget(QLabel("Pos:"))
        self.caption_position_combo = QComboBox()
        self.caption_position_combo.addItem("Top", "top")
        self.caption_position_combo.addItem("Mid", "center")
        self.caption_position_combo.addItem("Bot", "bottom")
        _cur_preset = get_preset(config.get_section("subtitles").get("style", "boxed_tiktok"))
        _pidx = self.caption_position_combo.findData(_cur_preset.get("position", "bottom"))
        if _pidx >= 0:
            self.caption_position_combo.setCurrentIndex(_pidx)
        self.caption_position_combo.setMaximumWidth(60)
        row_clr.addWidget(self.caption_position_combo)
        caption_layout.addLayout(row_clr)

        self.hinglish_check = QCheckBox("Hinglish (Hindi → Romanized)")
        self.hinglish_check.setToolTip("Convert Hindi/Devanagari captions to readable Hinglish text")
        caption_layout.addWidget(self.hinglish_check)

        # Live preview (compact)
        self.caption_preview = CaptionPreviewWidget()
        self.caption_preview.setMinimumHeight(100)
        caption_layout.addWidget(self.caption_preview, 1)

        # ---- Cinematic Effects ----------------------------------------------------
        effects_group = QGroupBox("Cinematic Effects")
        effects_layout = QVBoxLayout(effects_group)
        effects_layout.setSpacing(3)
        effects_layout.setContentsMargins(6, 6, 6, 6)

        grade_row = QHBoxLayout()
        grade_row.addWidget(QLabel("Color Grade:"))
        self.grade_combo = QComboBox()
        self.grade_combo.addItems(["None", "Warm", "Cool", "Teal/Orange", "Vintage", "Vibrant", "B&W"])
        self.grade_combo.setCurrentText("Warm")
        grade_row.addWidget(self.grade_combo)
        grade_row.addStretch()
        effects_layout.addLayout(grade_row)

        fx_grid = QGridLayout()
        fx_grid.setSpacing(4)
        self.fx_checks = {}
        self.fx_sliders = {}
        fx_items = [
            ("glow", "Glow"), ("grain", "Film Grain"), ("vignette", "Vignette"),
            ("sharpen", "Sharpen"), ("letterbox", "Letterbox"), ("chroma_shift", "Chroma Shift"),
            ("bottom_gradient", "Bot Gradient"), ("top_gradient", "Top Gradient"),
        ]
        for idx, (fx_id, fx_label) in enumerate(fx_items):
            row_idx = idx // 2
            col_idx = idx % 2
            sub = QHBoxLayout()
            cb = QCheckBox(fx_label)
            self.fx_checks[fx_id] = cb
            sub.addWidget(cb)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setEnabled(False)
            slider.setMaximumWidth(60)
            self.fx_sliders[fx_id] = slider
            sub.addWidget(slider)
            cb.toggled.connect(lambda checked, s=slider: s.setEnabled(checked))
            fx_grid.addLayout(sub, row_idx, col_idx)
        effects_layout.addLayout(fx_grid)

        # ---- 2-column options grid (sleek & compact) --------------------------
        options_grid = QGridLayout()
        options_grid.setSpacing(6)
        options_grid.setContentsMargins(0, 0, 0, 0)

        options_grid.addWidget(input_group, 0, 0)
        options_grid.addWidget(ai_group, 0, 1)
        options_grid.addWidget(caption_group, 1, 0)
        options_grid.addWidget(effects_group, 1, 1)

        # ---- Background Music ----------------------------------------------------
        music_group = QGroupBox("Background Music")
        music_layout = QVBoxLayout(music_group)
        music_layout.setSpacing(3)
        music_layout.setContentsMargins(6, 6, 6, 6)

        self.music_check = QCheckBox("Enable background music (sidechain ducking)")
        music_layout.addWidget(self.music_check)

        music_folder_row = QHBoxLayout()
        self.music_folder_label = QLabel("No folder selected")
        self.select_music_btn = QPushButton("Select Folder")
        music_folder_row.addWidget(self.music_folder_label, 1)
        music_folder_row.addWidget(self.select_music_btn)
        music_layout.addLayout(music_folder_row)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self.music_volume_slider = QSlider(Qt.Horizontal)
        self.music_volume_slider.setRange(0, 100)
        self.music_volume_slider.setValue(30)
        vol_row.addWidget(self.music_volume_slider, 1)
        self.music_volume_label = QLabel("30%")
        vol_row.addWidget(self.music_volume_label)
        music_layout.addLayout(vol_row)

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

        options_grid.addWidget(music_group, 2, 0, 1, 2)

        left_layout.addLayout(options_grid)

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
        preview_layout.setContentsMargins(6, 6, 6, 6)
        self.video_preview = VideoPreviewWidget()
        preview_layout.addWidget(self.video_preview)
        tabs.addTab(preview_tab, "Preview")

        right_layout.addWidget(tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([440, 760])

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

        # Live video preview — refresh whenever effects controls change.
        self.grade_combo.currentTextChanged.connect(self._update_effects_preview)
        for fx_cb in self.fx_checks.values():
            fx_cb.toggled.connect(self._update_effects_preview)
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
        if hasattr(self, "caption_preview"):
            style = self._current_caption_style()
            self.caption_preview.set_style(style)
            if hasattr(self, "video_preview"):
                self.video_preview.set_style(style)

    def _update_effects_preview(self, *_):
        if hasattr(self, "video_preview"):
            effects = {
                "grade": self.grade_combo.currentText(),
            }
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