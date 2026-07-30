import cv2
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from utils.logging import get_logger, JobLogger

logger = get_logger("reframer")


class AutoReframer:
    def __init__(self, shorts_config: Dict[str, Any], device: torch.device):
        self.config = shorts_config
        self.device = device
        self.target_width, self.target_height = shorts_config.get("target_resolution", [1080, 1920])
        self.face_tracking_enabled = shorts_config.get("face_tracking", {}).get("enabled", True)
        self.smoothing = shorts_config.get("face_tracking", {}).get("smoothing", 0.3)
        self.zoom_factor = shorts_config.get("face_tracking", {}).get("zoom_factor", 1.2)
        self.auto_zoom_enabled = shorts_config.get("auto_zoom", {}).get("enabled", True)
        self.zoom_range = shorts_config.get("auto_zoom", {}).get("zoom_range", [1.0, 1.5])
        self.padding = shorts_config.get("auto_reframe", {}).get("padding", 0.15)
        self.face_detector = None
        self._init_face_detector()

    def _init_face_detector(self):
        if not self.face_tracking_enabled:
            return
        # GPU face detection with automatic, crash-proof CPU fallback — see
        # utils/onnx_gpu.py and the note in video/analyzer.py for the cuDNN 8/9
        # collision this guards against. Real GPU when safe, CPU otherwise, never
        # a hard abort. Set shorts.face_tracking.force_cpu: true to force CPU.
        from utils.onnx_gpu import pick_providers
        force_cpu = bool(self.config.get("face_tracking", {}).get("force_cpu", False))
        providers, ctx_id, _cuda_ok = pick_providers(force_cpu=force_cpu)
        # CPU speedup: on machines without a usable GPU, shrink the detector
        # input and detect faces less often. self._cuda_ok is read by the reframe
        # loop to widen detect_interval. GPU path is unchanged.
        self._cuda_ok = _cuda_ok
        det_size = (640, 640) if _cuda_ok else tuple(
            self.config.get("face_tracking", {}).get("cpu_det_size", [480, 480])
        )
        try:
            from insightface.app import FaceAnalysis
            self.face_detector = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_detector.prepare(ctx_id=ctx_id, det_size=det_size)
            logger.info("Face detector loaded for reframing", cuda=_cuda_ok,
                        force_cpu=force_cpu, det_size=det_size)
        except Exception as e:
            logger.warning("Face detector not available for reframing", error=str(e))
            self.face_detector = None

    def reframe(
        self,
        input_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        job_logger: JobLogger
    ) -> Path:
        job_logger.info("Reframing clip", input=str(input_path), output=str(output_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_clip = output_path.parent / f"{output_path.stem}_temp.mp4"
        self._extract_clip(input_path, start_time, end_time, temp_clip, job_logger)

        if self.face_tracking_enabled and self.face_detector:
            self._process_with_reframing(temp_clip, output_path, job_logger)
        else:
            self._static_reframe(temp_clip, output_path, job_logger)

        if temp_clip.exists():
            temp_clip.unlink()

        return output_path

    def _extract_clip(self, input_path: Path, start: float, end: float, output: Path, job_logger: JobLogger):
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(input_path),
            "-t", str(end - start),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _static_reframe(self, input_path: Path, output_path: Path, job_logger: JobLogger):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf",
            f"crop=ih*9/16:ih,scale={self.target_width}:{self.target_height}",
            "-c:a", "copy",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _process_with_reframing(self, input_path: Path, output_path: Path, job_logger: JobLogger):
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_ratio = self.target_width / self.target_height
        source_ratio = width / height

        if source_ratio > target_ratio:
            crop_width = int(height * target_ratio)
            crop_height = height
        else:
            crop_width = width
            crop_height = int(width / target_ratio)

        temp_output = output_path.parent / f"{output_path.stem}_raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (self.target_width, self.target_height))

        tracked_center_x = width // 2
        tracked_center_y = height // 2
        prev_center_x = tracked_center_x
        prev_center_y = tracked_center_y
        frame_count = 0
        # Detect faces once per second on GPU; every ~2s on CPU (the tracked
        # center is held between detections, so motion stays smooth either way).
        base_interval = int(fps) if getattr(self, "_cuda_ok", True) else int(fps * 2)
        detect_interval = max(1, base_interval)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_count % detect_interval == 0:
                faces = self.face_detector.get(frame_rgb)

            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                center_x = int((face.bbox[0] + face.bbox[2]) / 2)
                center_y = int((face.bbox[1] + face.bbox[3]) / 2)

                tracked_center_x = int(self.smoothing * center_x + (1 - self.smoothing) * prev_center_x)
                tracked_center_y = int(self.smoothing * center_y + (1 - self.smoothing) * prev_center_y)
                prev_center_x = tracked_center_x
                prev_center_y = tracked_center_y

            half_crop_w = crop_width // 2
            half_crop_h = crop_height // 2

            x1 = max(0, tracked_center_x - half_crop_w)
            y1 = max(0, tracked_center_y - half_crop_h)
            x2 = min(width, x1 + crop_width)
            y2 = min(height, y1 + crop_height)

            if x2 - x1 < crop_width:
                if x1 == 0:
                    x2 = crop_width
                else:
                    x1 = width - crop_width
            if y2 - y1 < crop_height:
                if y1 == 0:
                    y2 = crop_height
                else:
                    y1 = height - crop_height

            cropped = frame[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (self.target_width, self.target_height))
            out.write(resized)
            frame_count += 1

        cap.release()
        out.release()

        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_output),
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if temp_output.exists():
            temp_output.unlink()

        job_logger.info("Reframing complete", frames=frame_count)