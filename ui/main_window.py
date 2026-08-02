from pathlib import Path
from typing import Optional, List, Callable
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QProgressBar, QListWidget, QTextEdit, QFileDialog,
    QGroupBox, QCheckBox, QTabWidget, QSplitter,
    QMessageBox, QSlider, QScrollArea, QLineEdit,
    QApplication, QColorDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDateTime, QRectF
from PySide6.QtGui import QFont, QIcon, QPixmap, QImage, QPainter, QPainterPath, QColor, QPen, QBrush, QFontMetrics, QFontDatabase
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


def _register_all_fonts():
    """Register bundled font files with Qt so QFont works in the preview.
    Must be called *after* QApplication is created (QFontDatabase needs it)."""
    from video.fonts import FONTS_DIR
    if FONTS_DIR.exists():
        for ttf in list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf")):
            QFontDatabase.addApplicationFont(str(ttf))


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

        # ── Animation-aware rendering ────────────────────────────────────
        animation = cfg.get("animation", "none")
        if animation == "one_word":
            lines = [[self.HIGHLIGHT_INDEX]]

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

        # Y-Offset on top of position-adjusted base.
        y_offset = int(cfg.get("y_offset", 0))
        if y_offset:
            y0 = y0 + (y_offset * scale)

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
                if animation in ("none", "karaoke"):
                    col = primary  # never highlight
                elif animation == "one_word":
                    col = highlight  # only one word shown
                else:
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

        # Draw animation badge at top of frame
        if animation and animation != "none":
            badge_font = QFont("Segoe UI", 7)
            p.setFont(badge_font)
            p.setPen(QColor(128, 128, 128, 180))
            p.drawText(frame.adjusted(4, 4, -4, -4), Qt.AlignTop | Qt.AlignRight, f"⚡ {animation}")

        # Note: NO end() here — QPainter owned by paintEvent caller


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
        self._anim_idx = 3
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._advance_animation)
        # Region blur + border config
        self._blur_cfg: dict = {}
        self._border_cfg: dict = {}
        self.setMinimumSize(280, 420)
        self.setStyleSheet("background: #1a1a1a; border-radius: 6px;")

    def _advance_animation(self):
        anim = self._cfg.get("animation", "none")
        if anim in ("highlight", "word_reveal", "one_word"):
            self._anim_idx = (self._anim_idx + 1) % len(self.SAMPLE_WORDS)
            self.update()

    def set_blur_config(self, cfg: dict):
        """Store region-blur config and refresh the preview."""
        self._blur_cfg = cfg or {}
        self.update()

    def set_border_config(self, cfg: dict):
        """Store border config and refresh the preview."""
        self._border_cfg = cfg or {}
        self.update()

    def set_video_path(self, path: Optional[Path]):
        self._frame_rgb = None
        if path and path.exists():
            try:
                import cv2
                cap = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid = max(0, total // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self._frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                self._frame_rgb = None
        self._frame = None  # unused — _frame_rgb replaces it
        self.update()

    def set_style(self, cfg: dict):
        self._cfg = cfg or {}
        self._anim_idx = 3  # reset
        anim = cfg.get("animation", "none") if cfg else "none"
        if anim and anim != "none":
            self._anim_timer.start(700)
        else:
            self._anim_timer.stop()
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

        if getattr(self, '_frame_rgb', None) is not None:
            fh, fw = self._frame_rgb.shape[:2]
            # Work on a copy so we don't corrupt the stored frame.
            display = self._frame_rgb.copy()
            self._apply_blur_effects(display, frame)
            frame_pix = QPixmap.fromImage(
                QImage(display.data, fw, fh, fw * 3, QImage.Format_RGB888)
            ).scaled(int(frame_w), int(frame_h), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            px = int(fx + (frame_w - frame_pix.width()) / 2)
            py = int(fy + (frame_h - frame_pix.height()) / 2)
            p.drawPixmap(px, py, frame_pix)
            # Border overlay around the 9:16 frame rectangle (not the pixmap,
            # since KeepAspectRatioByExpanding may crop edges of the pixmap).
            if self._border_cfg.get("enabled", False):
                bp = max(1, int(self._border_cfg.get("size", 4)))
                bc = self._border_cfg.get("color", "#FFFFFF")
                p.setPen(QPen(QColor(bc), bp))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(fx, fy, frame_w, frame_h))
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

        # ── Animation-aware rendering ────────────────────────────────────
        animation = cfg.get("animation", "none")
        highlight_idx = self._anim_idx if animation not in ("none", "karaoke") else self.HIGHLIGHT_INDEX
        if animation == "one_word":
            lines = [[self._anim_idx]]

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

        # Y-Offset applied on top of the position-adjusted base (same as ChangeGUI).
        y_offset = int(cfg.get("y_offset", 0))
        if y_offset:
            y0 = y0 + (y_offset * scale)

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
                if animation in ("none", "karaoke"):
                    col = primary
                elif animation == "one_word":
                    col = highlight
                else:
                    col = highlight if i == highlight_idx else primary
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

        # Draw animation badge at top of frame
        if animation and animation != "none":
            badge_font = QFont("Segoe UI", 7)
            p.setFont(badge_font)
            p.setPen(QColor(128, 128, 128, 180))
            p.drawText(frame.adjusted(4, 4, -4, -4), Qt.AlignTop | Qt.AlignRight, f"⚡ {animation}")

        # Note: NO end() here — QPainter owned by paintEvent caller

    # ── Region blur effects (applied to the frame COPY) ────────────────────
    def _apply_blur_effects(self, frame: np.ndarray, view_rect: QRectF):
        """Blur any combination of sides/center on the frame."""
        bc = self._blur_cfg
        if not bc.get("enabled", False):
            return
        h, w = frame.shape[:2]
        intensity = max(1, int(bc.get("intensity", 15)))
        tint_on = bc.get("tint_enabled", False)
        tint_hex = bc.get("tint_color", "#000000")
        tint_op = int(bc.get("tint_opacity", 50))

        # Build rects from per-side config
        sides = bc.get("sides", {})
        rects = []
        for side, sc in sides.items():
            if not sc.get("enabled", False):
                continue
            pct = max(0, min(100, int(sc.get("size", 30)))) / 100.0
            if side == "top":
                rects.append((0, 0, w, int(h * pct)))
            elif side == "bottom":
                rects.append((0, h - int(h * pct), w, h))
            elif side == "left":
                rects.append((0, 0, int(w * pct), h))
            elif side == "right":
                rects.append((w - int(w * pct), 0, w, h))
            elif side == "center":
                bw, bh = int(w * pct), int(h * pct)
                rects.append(((w - bw) // 2, (h - bh) // 2,
                              (w + bw) // 2, (h + bh) // 2))

        if not rects:
            return

        import cv2
        ksize = max(3, intensity // 3 * 2 + 1)  # odd kernel
        for x1, y1, x2, y2 in rects:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            if tint_on:
                try:
                    tc = tuple(int(tint_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                except Exception:
                    tc = (0, 0, 0)
                alpha = max(0, min(1.0, tint_op / 100.0))
                tinted = (blurred * (1 - alpha) + np.array(tc, dtype=np.uint8) * alpha).astype(np.uint8)
                frame[y1:y2, x1:x2] = tinted
            else:
                frame[y1:y2, x1:x2] = blurred

            # Draw outline around blur region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (59, 130, 246), max(2, w // 300))


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
    _SETTINGS_PATH = Path(__file__).parent.parent / "ui_state.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Viral Clipper")
        self.setMinimumSize(1200, 800)

        # Register bundled fonts with Qt (needs QApplication running).
        _register_all_fonts()

        self.job_manager = JobManager(Path("./data/jobs.db"))
        self.current_job: Optional[Job] = None
        self.worker: Optional[ProcessingWorker] = None
        self.clip_previews: List[str] = []

        self._setup_ui()
        self._load_app_settings()
        self._connect_signals()
        self._load_jobs_history()

    def closeEvent(self, event):
        self._save_app_settings()
        super().closeEvent(event)

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
        self.duration_spin.setRange(15, 1200)
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

        # ---- Whisper Model (transcription) ----------------------------------
        # Pick which speech-to-text model transcribes the source video. The
        # path label below shows where the model should be cached so the user
        # can verify it exists before processing.
        whisper_group = QGroupBox("Whisper Model (Transcription)")
        whisper_layout = QVBoxLayout(whisper_group)

        whisper_model_row = QHBoxLayout()
        whisper_model_row.addWidget(QLabel("Model:"))
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(
            ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"])
        self.whisper_model_combo.setCurrentText(
            config.get_section("whisper").get("model", "large-v3"))
        whisper_model_row.addWidget(self.whisper_model_combo, 1)
        whisper_layout.addLayout(whisper_model_row)

        self.whisper_path_label = QLabel()
        self.whisper_path_label.setWordWrap(True)
        self.whisper_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.whisper_path_label.setStyleSheet("color: #888; font-size: 11px;")
        whisper_layout.addWidget(self.whisper_path_label)

        self.whisper_model_combo.currentTextChanged.connect(self._on_whisper_model_changed)
        self._update_whisper_path_status()

        left_layout.addWidget(input_group)
        left_layout.addWidget(whisper_group)
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
        self.caption_color_btn = QPushButton()
        self.caption_color_btn.setFixedSize(28, 28)
        self.caption_color_btn.setToolTip("Click to pick primary color")
        self.caption_color_btn.setStyleSheet(
            "background: transparent; border: 1px dashed #666; border-radius: 3px;"
        )
        self.caption_color_btn.setText("P")
        self.caption_color_btn.clicked.connect(
            lambda: self._pick_color("_caption_color", self.caption_color_btn, "#FFFFFF")
        )
        self._caption_color = ""  # empty = use preset default
        cap_ov_row.addWidget(self.caption_color_btn)
        cap_ov_row.addWidget(QLabel("Highlight:"))
        self.highlight_color_btn = QPushButton()
        self.highlight_color_btn.setFixedSize(28, 28)
        self.highlight_color_btn.setToolTip("Click to pick highlight color")
        self.highlight_color_btn.setStyleSheet(
            "background: transparent; border: 1px dashed #666; border-radius: 3px;"
        )
        self.highlight_color_btn.setText("H")
        self.highlight_color_btn.clicked.connect(
            lambda: self._pick_color("_highlight_color", self.highlight_color_btn, "#FFD400")
        )
        self._highlight_color = ""  # empty = use preset default
        cap_ov_row.addWidget(self.highlight_color_btn)
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

        # Y-Offset: manual position adjustment (same as ChangeGUI's "caption_y_offset")
        yoff_row = QHBoxLayout()
        yoff_row.addWidget(QLabel("Y-Offset:"))
        self.y_offset_slider = QSlider(Qt.Horizontal)
        self.y_offset_slider.setRange(-200, 200)
        self.y_offset_slider.setValue(0)
        self.y_offset_slider.setFixedWidth(140)
        self.y_offset_slider.setToolTip("Adjust caption Y position (-200 to 200); applied on top of the position setting")
        yoff_row.addWidget(self.y_offset_slider)
        self.y_offset_label = QLabel("0")
        self.y_offset_label.setFixedWidth(30)
        yoff_row.addWidget(self.y_offset_label)
        caption_layout.addLayout(yoff_row)

        self.hinglish_check = QCheckBox("Hinglish transliteration (Hindi → Romanized)")
        self.hinglish_check.setToolTip("Convert Hindi/Devanagari captions to readable Hinglish text")
        caption_layout.addWidget(self.hinglish_check)

        # Live preview of the caption look (updates as controls change).
        caption_layout.addWidget(QLabel("Preview:"))
        self.caption_preview = CaptionPreviewWidget()
        caption_layout.addWidget(self.caption_preview, 1)

        # ── Caption control signals (connected here in _setup_ui so they always
        # fire, independent of _connect_signals / signal-reset timing).
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.font_combo.currentIndexChanged.connect(self._update_caption_preview)
        self.caption_size_spin.valueChanged.connect(self._update_caption_preview)
        self.caption_position_combo.currentIndexChanged.connect(self._on_preset_or_position_changed)
        self.y_offset_slider.valueChanged.connect(self._on_y_offset_changed)
        # Load the initial preset's defaults into the override controls so they show
        # its real look on startup (a saved session, if any, is restored afterward).
        self._on_preset_changed()

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

        # ---- Blur & Border tab ----------------------------------------------------
        bb_tab = QWidget()
        bb_layout = QHBoxLayout(bb_tab)
        bb_layout.setContentsMargins(4, 4, 4, 4)

        # Left: controls
        bb_controls = QWidget()
        bb_ctl_layout = QVBoxLayout(bb_controls)
        bb_ctl_layout.setContentsMargins(0, 0, 0, 0)

        self._build_blur_border_controls(bb_ctl_layout)

        # Right: preview
        self.bb_preview = VideoPreviewWidget()

        bb_layout.addWidget(bb_controls, 1)
        bb_layout.addWidget(self.bb_preview, 2)
        tabs.addTab(bb_tab, "Blur & Border")

        # Push initial blur/border config
        self._update_blur_border_preview()

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

        # Caption control signals are already connected in _setup_ui() — don't
        # duplicate them here. The color buttons also call _update_caption_preview
        # internally via _pick_color.

        # Live video preview — effects and aspect ratio changes.
        self.grade_combo.currentTextChanged.connect(self._update_effects_preview)
        for fx_cb in self.fx_checks.values():
            fx_cb.toggled.connect(self._update_effects_preview)
        self.aspect_combo.currentTextChanged.connect(
            lambda txt: (
                self.video_preview.set_aspect(txt),
                self.bb_preview.set_aspect(txt) if hasattr(self, "bb_preview") else None
            )
        )
        self._update_effects_preview()

    def _on_preset_changed(self, *_):
        """When a caption preset is selected, load ITS defaults into every override
        control (font, primary/highlight color, size, position) so the user sees the
        preset's real look. From there any manual change overrides it; leaving a
        control untouched keeps the preset default."""
        preset_id = self.preset_combo.currentData() or "boxed_tiktok"
        preset = get_preset(preset_id)

        # Block signals while we repopulate so this doesn't re-trigger repeatedly;
        # we call _update_caption_preview() once at the end.
        for w in (self.font_combo, self.caption_size_spin, self.caption_position_combo):
            w.blockSignals(True)

        # Font: select the preset's font if present in the list, else "(Preset default)".
        p_font = preset.get("font_family", "")
        if p_font:
            fidx = self.font_combo.findData(p_font)
            if fidx < 0:
                fidx = self.font_combo.findText(p_font)
            self.font_combo.setCurrentIndex(fidx if fidx >= 0 else 0)
        else:
            self.font_combo.setCurrentIndex(0)

        # Size: show the preset's actual size (was 0="preset"); user can still change it.
        p_size = int(preset.get("font_size", 0) or 0)
        self.caption_size_spin.setValue(p_size)

        # Position.
        p_pos = preset.get("position", "bottom")
        pidx = self.caption_position_combo.findData(p_pos)
        if pidx >= 0:
            self.caption_position_combo.setCurrentIndex(pidx)

        # Colors: load the preset's colors onto the swatch buttons so they're the
        # active values (and visibly reflect the preset).
        p_primary = preset.get("primary_color", "") or ""
        p_highlight = preset.get("highlight_color", "") or ""
        self._caption_color = p_primary
        self._highlight_color = p_highlight
        self._apply_color_button(self.caption_color_btn, p_primary, "P")
        self._apply_color_button(self.highlight_color_btn, p_highlight, "H")

        for w in (self.font_combo, self.caption_size_spin, self.caption_position_combo):
            w.blockSignals(False)

        self._update_caption_preview()

    def _apply_color_button(self, button: QPushButton, hex_str: str, placeholder: str):
        """Paint a swatch button with a color, or show its dashed placeholder when empty."""
        if hex_str:
            button.setStyleSheet(
                f"background: {hex_str}; border: 1px solid #555; border-radius: 3px; line-height: 1;"
            )
            button.setText("")
        else:
            button.setStyleSheet(
                "background: transparent; border: 1px dashed #666; border-radius: 3px;"
            )
            button.setText(placeholder)

    def _on_preset_or_position_changed(self, *_):
        self._update_caption_preview()

    def _on_y_offset_changed(self, value: int):
        self.y_offset_label.setText(str(value))
        self._update_caption_preview()

    def _current_caption_style(self) -> dict:
        """Merge the selected preset with the UI overrides — the same values the
        pipeline will send to the ASS engine — for the live preview."""
        preset_id = self.preset_combo.currentData() or "boxed_tiktok"
        cfg = dict(get_preset(preset_id))
        font_family = self.font_combo.currentData() or ""
        if font_family:
            cfg["font_family"] = font_family
        primary = getattr(self, '_caption_color', '').strip()
        if primary:
            cfg["primary_color"] = primary
        highlight = getattr(self, '_highlight_color', '').strip()
        if highlight:
            cfg["highlight_color"] = highlight
        size = self.caption_size_spin.value()
        if size > 0:
            cfg["font_size"] = size
        pos = self.caption_position_combo.currentData()
        if pos:
            cfg["position"] = pos
        cfg["y_offset"] = self.y_offset_slider.value()
        return cfg

    def _update_caption_preview(self, *_):
        style = self._current_caption_style()
        if hasattr(self, "caption_preview"):
            self.caption_preview.set_style(style)
        if hasattr(self, "video_preview"):
            self.video_preview.set_style(style)
        if hasattr(self, "bb_preview"):
            self.bb_preview.set_style(style)

    def _update_effects_preview(self, *_):
        if hasattr(self, "video_preview"):
            effects = {"grade": self.grade_combo.currentText()}
            for fx_id, cb in self.fx_checks.items():
                effects[fx_id] = cb.isChecked()
            self.video_preview.set_effects(effects)
        if hasattr(self, "bb_preview"):
            effects = {"grade": self.grade_combo.currentText()}
            for fx_id, cb in self.fx_checks.items():
                effects[fx_id] = cb.isChecked()
            self.bb_preview.set_effects(effects)

    def _pick_color(self, attr_name: str, button: QPushButton, default: str):
        """Open a QColorDialog and apply the chosen colour to preview + button."""
        stored = getattr(self, attr_name, "")
        initial = QColor(stored) if stored else QColor(default)
        color = QColorDialog.getColor(initial, self, "Pick Color")
        if color.isValid():
            hex_str = color.name()
            setattr(self, attr_name, hex_str)
            button.setStyleSheet(
                f"background: {hex_str}; border: 1px solid #555; border-radius: 3px; line-height: 1;"
            )
            button.setText("")
            self._update_caption_preview()
        else:
            # Keep the button in its "not yet picked" visual state
            pass

    # ── Blur & Border tab builder ───────────────────────────────────────────
    def _build_blur_border_controls(self, layout: QVBoxLayout):
        """Build the left-side controls for the Blur & Border tab."""

        # ── Region Blur card ───────────────────────────────────────────────
        blur_group = QGroupBox("Region Blur")
        bl = QVBoxLayout(blur_group)

        self._bb_enabled = QCheckBox("Enable Region Blur")
        bl.addWidget(self._bb_enabled)

        # Per-side checkboxes + size sliders
        self._bb_sides = {}  # side -> {"cb": QCheckBox, "sl": QSlider, "lb": QLabel}
        for side, label in [("top", "Top"), ("bottom", "Bottom"),
                            ("left", "Left"), ("right", "Right"), ("center", "Center")]:
            row = QHBoxLayout()
            cb = QCheckBox(label)
            row.addWidget(cb)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(5, 100)
            sl.setValue(30 if side in ("top", "bottom") else 20)
            sl.setMaximumWidth(120)
            row.addWidget(sl, 1)
            val_lb = QLabel(f"{sl.value()}%")
            val_lb.setMinimumWidth(32)
            row.addWidget(val_lb)
            row.addStretch()
            sl.valueChanged.connect(lambda v, lb=val_lb: lb.setText(f"{v}%"))
            self._bb_sides[side] = {"cb": cb, "sl": sl, "lb": val_lb}
            bl.addLayout(row)

        self._bb_intensity_slider = self._make_slider("Blur:", bl, 15, 1, 50, 1)

        # Tint
        tt = QHBoxLayout()
        self._bb_tint_cb = QCheckBox("Tint")
        tt.addWidget(self._bb_tint_cb)
        self._bb_tint_btn = QPushButton()
        self._bb_tint_btn.setFixedSize(24, 24)
        self._bb_tint_btn.setStyleSheet("background: #000000; border: 1px solid #555; border-radius: 3px;")
        self._bb_tint_btn.setToolTip("Pick tint color")
        self._bb_tint_btn.clicked.connect(lambda: self._pick_bb_color(
            "_bb_tint", self._bb_tint_btn, "#000000", self._update_blur_border_preview))
        self._bb_tint = "#000000"
        tt.addWidget(self._bb_tint_btn)
        tt.addWidget(QLabel("Opacity:"))
        self._bb_tint_op = QSlider(Qt.Horizontal)
        self._bb_tint_op.setRange(0, 100)
        self._bb_tint_op.setValue(50)
        tt.addWidget(self._bb_tint_op, 1)
        bl.addLayout(tt)

        layout.addWidget(blur_group)

        # ── Border card ────────────────────────────────────────────────────
        border_group = QGroupBox("Border")
        bdl = QVBoxLayout(border_group)

        self._bb_border_enabled = QCheckBox("Enable Border")
        bdl.addWidget(self._bb_border_enabled)

        bc = QHBoxLayout()
        bc.addWidget(QLabel("Color:"))
        self._bb_border_btn = QPushButton()
        self._bb_border_btn.setFixedSize(24, 24)
        self._bb_border_btn.setStyleSheet("background: #FFFFFF; border: 1px solid #555; border-radius: 3px;")
        self._bb_border_btn.setToolTip("Pick border color")
        self._bb_border_btn.clicked.connect(lambda: self._pick_bb_color(
            "_bb_border", self._bb_border_btn, "#FFFFFF", self._update_blur_border_preview))
        self._bb_border = "#FFFFFF"
        bc.addWidget(self._bb_border_btn)
        bc.addWidget(QLabel("Size:"))
        self._bb_border_size = QSpinBox()
        self._bb_border_size.setRange(1, 60)
        self._bb_border_size.setValue(4)
        self._bb_border_size.setSuffix("px")
        bc.addWidget(self._bb_border_size, 1)
        bdl.addLayout(bc)

        layout.addWidget(border_group)
        layout.addStretch()

        # Wire signals — all checkbox/slider changes refresh the preview
        self._bb_enabled.toggled.connect(self._update_blur_border_preview)
        for sd in self._bb_sides.values():
            sd["cb"].toggled.connect(self._update_blur_border_preview)
            sd["sl"].valueChanged.connect(self._update_blur_border_preview)
        self._bb_intensity_slider.valueChanged.connect(self._update_blur_border_preview)
        self._bb_tint_cb.toggled.connect(self._update_blur_border_preview)
        self._bb_tint_op.valueChanged.connect(self._update_blur_border_preview)
        self._bb_border_enabled.toggled.connect(self._update_blur_border_preview)
        self._bb_border_size.valueChanged.connect(self._update_blur_border_preview)

    def _make_slider(self, label: str, parent_layout, default: int,
                     lo: int, hi: int, step: int = 1) -> QSlider:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(default)
        sl.setSingleStep(step)
        row.addWidget(sl, 1)
        sl._label_hack = QLabel(f"{sl.value()}")
        row.addWidget(sl._label_hack)
        sl.valueChanged.connect(lambda v: sl._label_hack.setText(str(v)))
        parent_layout.addLayout(row)
        return sl

    def _pick_bb_color(self, attr: str, btn: QPushButton, default: str, after):
        stored = getattr(self, attr, "")
        initial = QColor(stored) if stored else QColor(default)
        c = QColorDialog.getColor(initial, self, "Pick Color")
        if c.isValid():
            setattr(self, attr, c.name())
            btn.setStyleSheet(
                f"background: {c.name()}; border: 1px solid #555; border-radius: 3px;"
            )
            btn.setText("")
            after()

    def _update_blur_border_preview(self, *_):
        """Read the blur/border controls and push to both preview widgets."""
        sides = {}
        for sd_name, sd in self._bb_sides.items():
            sides[sd_name] = {
                "enabled": sd["cb"].isChecked(),
                "size": sd["sl"].value(),
            }
        blur_cfg = {
            "enabled": self._bb_enabled.isChecked(),
            "sides": sides,
            "intensity": self._bb_intensity_slider.value(),
            "tint_enabled": self._bb_tint_cb.isChecked(),
            "tint_color": getattr(self, "_bb_tint", "#000000"),
            "tint_opacity": self._bb_tint_op.value(),
        }
        border_cfg = {
            "enabled": self._bb_border_enabled.isChecked(),
            "color": getattr(self, "_bb_border", "#FFFFFF"),
            "size": self._bb_border_size.value(),
        }
        if hasattr(self, "video_preview"):
            self.video_preview.set_blur_config(blur_cfg)
            self.video_preview.set_border_config(border_cfg)
        if hasattr(self, "bb_preview"):
            self.bb_preview.set_blur_config(blur_cfg)
            self.bb_preview.set_border_config(border_cfg)

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
            if hasattr(self, "bb_preview"):
                self.bb_preview.set_video_path(self.video_path)
            self._save_app_settings()

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path = Path(path)
            self.output_label.setText(f"Output: {path}")
            self._save_app_settings()

    def _select_music_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Background Music Folder")
        if path:
            self.music_folder = Path(path)
            self.music_folder_label.setText(self.music_folder.name)
            self.music_check.setChecked(True)
            logger.info("Music folder selected", path=path)
            self._save_app_settings()

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
        caption_color = getattr(self, '_caption_color', "") or ""
        highlight_color = getattr(self, '_highlight_color', "") or ""
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
                "caption_y_offset": self.y_offset_slider.value(),
                "hinglish": self.hinglish_check.isChecked(),
                "effects": effects_cfg if effects_cfg else None,
                "music": music_cfg,
                # Region blur + border config (built fresh from UI controls)
                "blur": {
                    "enabled": self._bb_enabled.isChecked(),
                    "sides": {
                        sd_name: {
                            "enabled": sd["cb"].isChecked(),
                            "size": sd["sl"].value(),
                        }
                        for sd_name, sd in self._bb_sides.items()
                    },
                    "intensity": self._bb_intensity_slider.value(),
                    "tint_enabled": self._bb_tint_cb.isChecked(),
                    "tint_color": getattr(self, "_bb_tint", "#000000"),
                    "tint_opacity": self._bb_tint_op.value(),
                },
                "border": {
                    "enabled": self._bb_border_enabled.isChecked(),
                    "color": getattr(self, "_bb_border", "#FFFFFF"),
                    "size": self._bb_border_size.value(),
                },
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
        self.caption_color_btn.setEnabled(not running)
        self.highlight_color_btn.setEnabled(not running)
        self.caption_size_spin.setEnabled(not running)
        self.caption_position_combo.setEnabled(not running)
        self.y_offset_slider.setEnabled(not running)
        self.grade_combo.setEnabled(not running)
        for cb in self.fx_checks.values():
            cb.setEnabled(not running)
        self.music_check.setEnabled(not running)
        self.select_music_btn.setEnabled(not running)
        self.music_volume_slider.setEnabled(not running)
        self.music_duck_slider.setEnabled(not running)
        self.music_random_check.setEnabled(not running)
        self.hinglish_check.setEnabled(not running)
        self.whisper_model_combo.setEnabled(not running)

    # ---- Whisper model helpers ------------------------------------------------
    def _on_whisper_model_changed(self, model: str):
        """Persist the chosen whisper model to settings.yaml (pipeline reads it)."""
        if not model:
            return
        config.update("whisper", "model", model)
        try:
            config.save()
        except Exception as e:
            logger.warning("Could not save whisper model setting", error=str(e))
        self._update_whisper_path_status()

    def _whisper_cache_paths(self, model: str):
        """Return (local_dirs, hf_cache) where a whisper model may be downloaded.

        Mirrors ``ai/transcriber.py``: the local cache lives under
        ``models/whisper/`` in HF-cache layout (``models--<repo-with-dashes>``)
        or a flat folder. Missing models fall back to the default HuggingFace
        cache under the user profile.
        """
        if model.startswith("distil-"):
            hf_id = f"Systran/faster-distil-whisper-{model[len('distil-'):]}"
        else:
            hf_id = f"Systran/faster-whisper-{model}"
        folder = hf_id.replace("/", "--")
        base = Path(__file__).resolve().parent.parent / "models" / "whisper"
        local_dirs = [base / f"models--{folder}", base / folder]
        hf = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{folder}"
        return local_dirs, hf

    def _update_whisper_path_status(self):
        """Refresh the whisper-model path label: ready vs. will-download."""
        if not hasattr(self, "whisper_model_combo"):
            return
        model = self.whisper_model_combo.currentText()
        local_dirs, hf = self._whisper_cache_paths(model)

        def _has_model(d: Path) -> bool:
            return (d / "model.bin").is_file() or bool(list(d.rglob("model.bin")))

        found = next((d for d in local_dirs if _has_model(d)), None)
        if found is not None:
            text = f"✅ {model} ready:\n{found}\\model.bin"
        elif _has_model(hf):
            text = f"✅ {model} in HF cache:\n{hf}"
        else:
            text = (f"⚠️ {model} not downloaded — will download on first run.\n"
                    f"  Local: {local_dirs[0]}\\model.bin\n"
                    f"  HF cache: {hf}")
        self.whisper_path_label.setText(text)

    # ---- Settings persistence -------------------------------------------------
    def _save_app_settings(self):
        """Save all UI controls state to a JSON file for next launch."""
        import json
        data = {
            # Input / output
            "video_path": str(getattr(self, "video_path", "")) if hasattr(self, "video_path") else "",
            "output_dir": str(getattr(self, "output_path", "./output")),
            # Processing params
            "num_clips": self.clips_spin.value(),
            "clip_duration": self.duration_spin.value(),
            "style": self.style_combo.currentText(),
            "subtitles": self.subtitle_check.isChecked(),
            "aspect_ratio": self.aspect_combo.currentText(),
            # AI ranking
            "llm_enabled": self.llm_check.isChecked(),
            "llm_provider": self.provider_combo.currentText(),
            "llm_model": self.model_combo.currentText().strip(),
            # Caption style
            "caption_preset": self.preset_combo.currentData() or "",
            "caption_font": self.font_combo.currentData() or "",
            "caption_color": getattr(self, "_caption_color", ""),
            "highlight_color": getattr(self, "_highlight_color", ""),
            "caption_size": self.caption_size_spin.value(),
            "caption_position": self.caption_position_combo.currentData() or "",
            "caption_y_offset": self.y_offset_slider.value(),
            "hinglish": self.hinglish_check.isChecked(),
            # Cinematic effects
            "color_grade": self.grade_combo.currentText(),
            "fx": {
                fx_id: {
                    "enabled": cb.isChecked(),
                    "strength": self.fx_sliders[fx_id].value(),
                }
                for fx_id, cb in self.fx_checks.items()
            },
            # Music
            "music_enabled": self.music_check.isChecked(),
            "music_folder": str(getattr(self, "music_folder", "")) if hasattr(self, "music_folder") else "",
            "music_volume": self.music_volume_slider.value(),
            "music_duck": self.music_duck_slider.value(),
            "music_random": self.music_random_check.isChecked(),
            # Blur
            "blur_enabled": self._bb_enabled.isChecked(),
            "blur_sides": {
                sd_name: {
                    "enabled": sd["cb"].isChecked(),
                    "size": sd["sl"].value(),
                }
                for sd_name, sd in self._bb_sides.items()
            },
            "blur_intensity": self._bb_intensity_slider.value(),
            "blur_tint_enabled": self._bb_tint_cb.isChecked(),
            "blur_tint_color": getattr(self, "_bb_tint", "#000000"),
            "blur_tint_opacity": self._bb_tint_op.value(),
            # Border
            "border_enabled": self._bb_border_enabled.isChecked(),
            "border_color": getattr(self, "_bb_border", "#FFFFFF"),
            "border_size": self._bb_border_size.value(),
        }
        try:
            with open(self._SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # best effort

    def _load_app_settings(self):
        """Restore all UI controls from the saved JSON file."""
        import json
        if not self._SETTINGS_PATH.exists():
            return
        try:
            with open(self._SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        # Input / output
        vp = data.get("video_path", "")
        if vp and Path(vp).exists():
            self.video_path = Path(vp)
            self.video_label.setText(Path(vp).name)
            if hasattr(self, "video_preview"):
                self.video_preview.set_video_path(self.video_path)
            if hasattr(self, "bb_preview"):
                self.bb_preview.set_video_path(self.video_path)
        op = data.get("output_dir", "")
        if op:
            self.output_path = Path(op)
            self.output_label.setText(f"Output: {op}")

        # Processing params
        self.clips_spin.setValue(data.get("num_clips", 10))
        self.duration_spin.setValue(data.get("clip_duration", 30))
        style_txt = data.get("style", "")
        if style_txt:
            idx = self.style_combo.findText(style_txt)
            if idx >= 0:
                self.style_combo.setCurrentIndex(idx)
        self.subtitle_check.setChecked(data.get("subtitles", True))
        aspect_txt = data.get("aspect_ratio", "")
        if aspect_txt:
            idx = self.aspect_combo.findText(aspect_txt)
            if idx >= 0:
                self.aspect_combo.setCurrentIndex(idx)

        # AI ranking
        self.llm_check.setChecked(data.get("llm_enabled", False))
        prov = data.get("llm_provider", "")
        if prov:
            idx = self.provider_combo.findText(prov)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
        model_txt = data.get("llm_model", "")
        if model_txt:
            self.model_combo.setCurrentText(model_txt)

        # Caption style — block signals to avoid triggering preview refresh per-item
        self.preset_combo.blockSignals(True)
        preset = data.get("caption_preset", "")
        if preset:
            idx = self.preset_combo.findData(preset)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

        self.font_combo.blockSignals(True)
        font = data.get("caption_font", "")
        if font:
            idx = self.font_combo.findText(font)
            if idx >= 0:
                self.font_combo.setCurrentIndex(idx)
        self.font_combo.blockSignals(False)

        cap_color = data.get("caption_color", "")
        if cap_color:
            self._caption_color = cap_color
            self.caption_color_btn.setStyleSheet(
                f"background: {cap_color}; border: 1px solid #555; border-radius: 3px;"
            )
            self.caption_color_btn.setText("")
        hl_color = data.get("highlight_color", "")
        if hl_color:
            self._highlight_color = hl_color
            self.highlight_color_btn.setStyleSheet(
                f"background: {hl_color}; border: 1px solid #555; border-radius: 3px;"
            )
            self.highlight_color_btn.setText("")

        self.caption_size_spin.setValue(data.get("caption_size", 0))
        cap_pos = data.get("caption_position", "")
        if cap_pos:
            idx = self.caption_position_combo.findData(cap_pos)
            if idx >= 0:
                self.caption_position_combo.setCurrentIndex(idx)
        self.y_offset_slider.setValue(data.get("caption_y_offset", 0))
        self.hinglish_check.setChecked(data.get("hinglish", False))

        # Cinematic effects
        grade = data.get("color_grade", "")
        if grade:
            idx = self.grade_combo.findText(grade)
            if idx >= 0:
                self.grade_combo.setCurrentIndex(idx)
        fx_data = data.get("fx", {})
        for fx_id, state in fx_data.items():
            if fx_id in self.fx_checks:
                self.fx_checks[fx_id].setChecked(state.get("enabled", False))
                self.fx_sliders[fx_id].setValue(state.get("strength", 50))

        # Music
        self.music_check.setChecked(data.get("music_enabled", False))
        mf = data.get("music_folder", "")
        if mf and Path(mf).is_dir():
            self.music_folder = Path(mf)
            self.music_folder_label.setText(Path(mf).name)
        self.music_volume_slider.setValue(data.get("music_volume", 30))
        self.music_duck_slider.setValue(data.get("music_duck", 50))
        self.music_random_check.setChecked(data.get("music_random", True))

        # Blur
        self._bb_enabled.setChecked(data.get("blur_enabled", False))
        blur_sides = data.get("blur_sides", {})
        for sd_name, state in blur_sides.items():
            if sd_name in self._bb_sides:
                self._bb_sides[sd_name]["cb"].setChecked(state.get("enabled", False))
                self._bb_sides[sd_name]["sl"].setValue(state.get("size", 30))
        self._bb_intensity_slider.setValue(data.get("blur_intensity", 15))
        self._bb_tint_cb.setChecked(data.get("blur_tint_enabled", False))
        tc = data.get("blur_tint_color", "")
        if tc:
            self._bb_tint = tc
            self._bb_tint_btn.setStyleSheet(
                f"background: {tc}; border: 1px solid #555; border-radius: 3px;"
            )
        self._bb_tint_op.setValue(data.get("blur_tint_opacity", 50))

        # Border
        self._bb_border_enabled.setChecked(data.get("border_enabled", False))
        bc = data.get("border_color", "")
        if bc:
            self._bb_border = bc
            self._bb_border_btn.setStyleSheet(
                f"background: {bc}; border: 1px solid #555; border-radius: 3px;"
            )
        self._bb_border_size.setValue(data.get("border_size", 4))

        # Push restored blur/border to preview widgets
        self._update_blur_border_preview()
        # Refresh caption preview with restored values
        self._update_caption_preview()

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