import cv2
import numpy as np
import os
from typing import List, Dict, Any
from pathlib import Path
import torch
from core.models import TranscriptSegment, VideoFeatures
from utils.logging import get_logger, JobLogger
from utils.gpu import get_device

logger = get_logger("video_analyzer")


class VideoAnalyzer:
    """Visual feature extraction.

    Old design seeked the VideoCapture (cap.set(POS_FRAMES)) once per sampled
    frame *per segment*. With overlapping candidate windows that is thousands of
    random seeks over the same file — pathologically slow, especially on AV1/HEVC
    where each seek triggers a keyframe re-decode.

    New design: decode the video ONCE, linearly, sampling at a fixed time stride.
    Per-frame features (faces, motion, scene-change, sharpness) are collected on a
    timeline, then aggregated onto each candidate window by time overlap. Cost is
    now O(video_length / stride), independent of how many overlapping candidates
    exist.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.face_detector = None
        self.emotion_model = None
        self.bg_subtractor = None
        # Sample ~2 frames/sec by default — enough for faces/motion/scene signals.
        self.sample_interval = float(config.get("sample_interval", 0.5))
        self.scene_threshold = config.get("scene_detection", {}).get("threshold", 0.3)
        self._init_models()

    def _init_models(self):
        face_config = self.config.get("face_detection", {})
        model_name = face_config.get("model", "buffalo_l")
        det_size = face_config.get("det_size", [640, 640])

        # GPU face detection with automatic, crash-proof CPU fallback.
        #
        # The pipeline transcribes with faster-whisper (ctranslate2) on the GPU
        # FIRST, which loads a cuDNN 9 DLL into the process. onnxruntime-gpu then
        # needs cudnnGetLibConfig from that same cuDNN 9 — and if the resident
        # copy is an older 9.x that lacks the symbol, ORT hard-aborts the whole
        # process ("Could not load symbol cudnnGetLibConfig. Error code 127").
        # utils.onnx_gpu.pick_providers() probes for exactly that and returns CPU
        # when the GPU isn't safe, so we get real GPU acceleration on a properly
        # configured machine and a graceful CPU fallback everywhere else — no
        # crash, no cuDNN download. Set video.face_detection.force_cpu: true to
        # skip the probe and always use CPU.
        from utils.onnx_gpu import pick_providers
        force_cpu = bool(face_config.get("force_cpu", False))
        providers, ctx_id, _cuda_ok = pick_providers(force_cpu=force_cpu)

        # CPU speedup (machines without a usable GPU): face detection dominates
        # analysis wall-time on CPU. Shrink the detector input and sample fewer
        # frames when we're NOT on CUDA. On GPU nothing changes, so quality on a
        # properly configured machine is identical. Both are overridable via
        # video.face_detection.cpu_det_size / video.cpu_sample_interval.
        if not _cuda_ok:
            cpu_det = face_config.get("cpu_det_size", [480, 480])
            # never upscale past the configured GPU det_size
            det_size = [min(det_size[0], cpu_det[0]), min(det_size[1], cpu_det[1])]
            cpu_interval = float(self.config.get("cpu_sample_interval", 2.0))
            if cpu_interval > self.sample_interval:
                self.sample_interval = cpu_interval

        try:
            from insightface.app import FaceAnalysis
            self.face_detector = FaceAnalysis(name=model_name, providers=providers)
            self.face_detector.prepare(ctx_id=ctx_id, det_size=tuple(det_size))
            logger.info(
                "InsightFace loaded", model=model_name, cuda=_cuda_ok,
                providers=providers, force_cpu=force_cpu,
                det_size=det_size, sample_interval=self.sample_interval,
            )
        except Exception as e:
            logger.warning("InsightFace not available, using OpenCV", error=str(e))
            self.face_detector = None

        emotion_config = self.config.get("emotion_detection", {})
        if emotion_config.get("enabled", True):
            try:
                from fer.fer import FER
                self.emotion_model = FER(mtcnn=True)
                logger.info("FER emotion detector loaded")
            except Exception as e:
                logger.warning("FER not available", error=str(e))

        motion_config = self.config.get("motion_detection", {})
        if motion_config.get("enabled", True):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=motion_config.get("history", 500),
                varThreshold=motion_config.get("var_threshold", 16)
            )

    def analyze(
        self,
        video_path: Path,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger
    ) -> List[VideoFeatures]:
        job_logger.info("Analyzing video (single-pass)", video_path=str(video_path))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(round(fps * self.sample_interval)))

        # Per-sample timeline collected in a single forward pass.
        times: List[float] = []
        faces_timeline: List[int] = []
        emotions_timeline: List[Dict[str, float]] = []
        motion_timeline: List[float] = []
        scene_timeline: List[bool] = []
        sharp_timeline: List[float] = []

        prev_gray = None
        frame_idx = 0
        sampled = 0

        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                t = frame_idx / fps
                self._process_frame(
                    frame, t, prev_gray,
                    times, faces_timeline, emotions_timeline,
                    motion_timeline, scene_timeline, sharp_timeline
                )
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sampled += 1
            frame_idx += 1

        cap.release()
        job_logger.info("Video decoded", frames_sampled=sampled, fps=round(fps, 2))

        times_arr = np.array(times) if times else np.array([0.0])
        features = [
            self._aggregate_segment(
                segment, times_arr,
                faces_timeline, emotions_timeline,
                motion_timeline, scene_timeline, sharp_timeline
            )
            for segment in transcript
        ]

        job_logger.info("Video analysis complete", segments=len(features))
        return features

    def _process_frame(
        self, frame, t, prev_gray,
        times, faces_timeline, emotions_timeline,
        motion_timeline, scene_timeline, sharp_timeline
    ):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        n_faces = 0
        face_boxes = []
        if self.face_detector:
            try:
                faces = self.face_detector.get(frame_rgb)
                n_faces = len(faces)
                face_boxes = [f.bbox.astype(int).tolist() for f in faces]
            except Exception:
                pass

        emo = {}
        if self.emotion_model and face_boxes:
            for box in face_boxes:
                x1, y1, x2, y2 = box
                roi = frame_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                    try:
                        det = self.emotion_model.detect_emotions(roi)
                        if det:
                            emo = det[0]["emotions"]
                            break
                    except Exception:
                        pass

        motion = 0.0
        if self.bg_subtractor is not None:
            fg = self.bg_subtractor.apply(frame_gray)
            motion = float(np.sum(fg > 0) / fg.size)

        scene = False
        if prev_gray is not None:
            diff = cv2.absdiff(frame_gray, prev_gray)
            if (np.mean(diff) / 255.0) > self.scene_threshold:
                scene = True

        sharp = float(np.var(cv2.Laplacian(frame_gray, cv2.CV_64F))) / 10000.0

        times.append(t)
        faces_timeline.append(n_faces)
        emotions_timeline.append(emo)
        motion_timeline.append(motion)
        scene_timeline.append(scene)
        sharp_timeline.append(sharp)

    def _aggregate_segment(
        self, segment: TranscriptSegment, times_arr: np.ndarray,
        faces_timeline, emotions_timeline,
        motion_timeline, scene_timeline, sharp_timeline
    ) -> VideoFeatures:
        lo = int(np.searchsorted(times_arr, segment.start, side="left"))
        hi = int(np.searchsorted(times_arr, segment.end, side="right"))
        lo = max(0, min(lo, len(times_arr) - 1))
        hi = max(lo + 1, min(hi, len(times_arr)))

        if not faces_timeline:
            return VideoFeatures(
                segment_start=segment.start, segment_end=segment.end, faces_detected=0
            )

        faces = faces_timeline[lo:hi] or [0]
        motions = motion_timeline[lo:hi] or [0.0]
        scenes = scene_timeline[lo:hi]
        sharps = sharp_timeline[lo:hi] or [0.0]
        emos = [e for e in emotions_timeline[lo:hi] if e]

        avg_emotions = {}
        if emos:
            for key in emos[0].keys():
                avg_emotions[key] = float(np.mean([e[key] for e in emos]))

        return VideoFeatures(
            segment_start=segment.start,
            segment_end=segment.end,
            faces_detected=int(max(faces)),
            face_boxes=[],
            emotions=avg_emotions,
            motion_intensity=float(np.mean(motions)),
            scene_change=any(scenes),
            visual_activity_score=float(np.mean(sharps)),
        )
