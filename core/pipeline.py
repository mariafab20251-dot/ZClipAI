from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import time
from .models import Job, JobStatus, Clip, TranscriptSegment, ScoredSegment
from .job_manager import JobManager
from utils.logging import get_logger, JobLogger
from utils.cache import CacheManager, NumpyCacheManager
from utils.gpu import GPUManager, empty_cache
from config import config

logger = get_logger("pipeline")


class Pipeline:
    def __init__(
        self,
        job_manager: JobManager,
        cache_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.job_manager = job_manager
        self.cache = CacheManager(cache_dir / "pipeline")
        self.numpy_cache = NumpyCacheManager(cache_dir / "numpy")
        self.gpu = GPUManager()
        self.progress_callback = progress_callback
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _update_progress(self, job: Job, step: str, progress: float):
        job.progress = progress
        job.current_step = step
        self.job_manager.save_job(job)
        if self.progress_callback:
            self.progress_callback(step, progress)
        logger.info("Progress", job_id=job.id, step=step, progress=progress)

    def run(self, job: Job) -> List[Clip]:
        self._cancelled = False
        job_logger = JobLogger(job.id)

        try:
            self._update_progress(job, "extracting_audio", 0.05)
            audio_path = self._extract_audio(job, job_logger)

            self._update_progress(job, "transcribing", 0.15)
            transcript = self._transcribe(job, audio_path, job_logger)

            self._update_progress(job, "generating_candidates", 0.25)
            candidates = self._generate_candidates(job, transcript, job_logger)

            # All downstream analysis operates on candidate clips, not raw ASR
            # segments. The three analyses are independent, so run them concurrently.
            self._update_progress(job, "analyzing", 0.30)
            transcript_analysis, audio_features, video_features = self._analyze_all(
                job, audio_path, candidates, job_logger
            )

            self._update_progress(job, "scoring", 0.70)
            scored_segments = self._score_segments(
                job, candidates, transcript_analysis, audio_features, video_features, job_logger
            )

            # Optional LLM re-ranking. No-ops (returns heuristic order) when
            # disabled or no API key is present, so this never blocks a run.
            self._update_progress(job, "reranking", 0.78)
            scored_segments = self._rerank_segments(job, scored_segments, job_logger)

            self._update_progress(job, "selecting_clips", 0.80)
            selected_clips = self._select_clips(job, scored_segments, job_logger)

            self._update_progress(job, "reframing", 0.90)
            final_clips = self._reframe_and_export(job, selected_clips, job_logger)

            self._update_progress(job, "completed", 1.0)
            job.status = JobStatus.COMPLETED
            self.job_manager.save_job(job)
            job_logger.info("Pipeline completed successfully", clips=len(final_clips))
            return final_clips

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            self.job_manager.save_job(job)
            import traceback
            traceback.print_exc()
            job_logger.error("Pipeline failed", error=str(e))
            raise

    def _extract_audio(self, job: Job, job_logger: JobLogger) -> Path:
        from audio.extractor import AudioExtractor
        extractor = AudioExtractor(self.gpu.device)
        output_path = job.output_dir / f"{job.id}_audio.wav"
        return extractor.extract(job.input_video, output_path, job_logger)

    def _transcribe(self, job: Job, audio_path: Path, job_logger: JobLogger) -> List[TranscriptSegment]:
        from ai.transcriber import Transcriber
        transcriber = Transcriber(config.get_section("whisper"))
        cache_key = f"transcript_{job.id}_{audio_path.stat().st_mtime}"
        job_logger.info("Transcribing audio")
        return self.cache.get_or_compute(
            cache_key,
            transcriber.transcribe,
            audio_path,
            job_logger
        )

    def _generate_candidates(
        self,
        job: Job,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger
    ) -> List[TranscriptSegment]:
        from ai.candidate_generator import CandidateGenerator
        generator = CandidateGenerator(config.get_section("candidates"))
        cache_key = f"candidates_{job.id}"
        return self.cache.get_or_compute(
            cache_key,
            generator.generate,
            transcript,
            job_logger,
            job.clip_duration,
        )

    def _analyze_all(
        self,
        job: Job,
        audio_path: Path,
        candidates: List[TranscriptSegment],
        job_logger: JobLogger
    ):
        """Run the three independent analyses concurrently. Each is index-aligned
        to `candidates`. Video analysis is the slowest, so it runs in its own
        thread while audio/transcript proceed."""
        from concurrent.futures import ThreadPoolExecutor

        results: Dict[str, Any] = {}
        errors: Dict[str, Exception] = {}

        def run(name, fn):
            try:
                results[name] = fn()
            except Exception as e:
                errors[name] = e
                job_logger.error(f"{name} analysis failed", error=str(e))

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [
                ex.submit(run, "transcript", lambda: self._analyze_transcript(job, candidates, job_logger)),
                ex.submit(run, "audio", lambda: self._analyze_audio(job, audio_path, candidates, job_logger)),
                ex.submit(run, "video", lambda: self._analyze_video(job, job.input_video, candidates, job_logger)),
            ]
            for f in futures:
                f.result()

        if "transcript" in errors:
            raise errors["transcript"]
        # Audio/video degrade gracefully — scorer already handles None entries.
        transcript_analysis = results.get("transcript", {"segments": [], "global_stats": {}})
        audio_features = results.get("audio", [None] * len(candidates))
        video_features = results.get("video", [None] * len(candidates))
        return transcript_analysis, audio_features, video_features

    def _analyze_transcript(
        self,
        job: Job,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger
    ) -> Dict[str, Any]:
        from ai.transcript_analyzer import TranscriptAnalyzer
        analyzer = TranscriptAnalyzer()
        cache_key = f"transcript_analysis_{job.id}"
        return self.cache.get_or_compute(
            cache_key,
            analyzer.analyze,
            transcript,
            job_logger
        )

    def _analyze_audio(
        self,
        job: Job,
        audio_path: Path,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger
    ) -> List:
        from audio.analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer(config.get_section("audio"))
        cache_key = f"audio_features_{job.id}_{audio_path.stat().st_mtime}"
        return self.numpy_cache.get_or_compute(
            cache_key,
            analyzer.analyze,
            audio_path,
            transcript,
            job_logger
        )

    def _analyze_video(
        self,
        job: Job,
        video_path: Path,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger
    ) -> List:
        from video.analyzer import VideoAnalyzer
        analyzer = VideoAnalyzer(config.get_section("video"), self.gpu.device)
        cache_key = f"video_features_{job.id}_{video_path.stat().st_mtime}"
        return self.numpy_cache.get_or_compute(
            cache_key,
            analyzer.analyze,
            video_path,
            transcript,
            job_logger
        )

    def _score_segments(
        self,
        job: Job,
        transcript: List[TranscriptSegment],
        transcript_analysis: Dict,
        audio_features: List,
        video_features: List,
        job_logger: JobLogger
    ) -> List[ScoredSegment]:
        from ai.scorer import ViralScorer
        scorer = ViralScorer(config.get_section("scoring"))
        cache_key = f"scored_segments_{job.id}"
        return self.cache.get_or_compute(
            cache_key,
            scorer.score,
            transcript,
            transcript_analysis,
            audio_features,
            video_features,
            job_logger
        )

    def _rerank_segments(
        self,
        job: Job,
        scored_segments: List[ScoredSegment],
        job_logger: JobLogger
    ) -> List[ScoredSegment]:
        """Optionally blend an LLM's viral judgment into the heuristic ranking.

        Per-job overrides from the UI (job.config['llm']) take precedence over the
        global settings.yaml block. Intentionally NOT cached: keys/config can
        change between runs and the call is cheap relative to transcription.
        """
        from ai.llm_reranker import LLMReranker

        llm_cfg = dict(config.get_section("llm"))
        job_override = job.config.get("llm") if isinstance(job.config, dict) else None
        if isinstance(job_override, dict):
            llm_cfg.update({k: v for k, v in job_override.items() if v is not None})

        reranker = LLMReranker(llm_cfg)
        active, reason = reranker.is_active()
        if not active:
            job_logger.info("LLM rerank skipped", reason=reason)
            return scored_segments

        result = reranker.rerank(scored_segments, job_logger)
        # Log the effect for the Processing Log tab so the user can see it.
        judged = [s for s in result if s.llm_score is not None]
        if judged:
            job_logger.info(
                "AI re-ranked clips",
                judged=len(judged),
                top_changes=[
                    f"#{s.segment.id} LLM={s.llm_score:.0f} Heur={s.heuristic_score:.0f}"
                    for s in result[:5]
                ],
            )
        return result

    def _select_clips(
        self,
        job: Job,
        scored_segments: List[ScoredSegment],
        job_logger: JobLogger
    ) -> List[Clip]:
        from ai.segment_selector import SegmentSelector
        selector = SegmentSelector(
            config.get_section("selection"),
            config.get_section("scoring.thresholds"),
            target_duration=job.clip_duration
        )
        return selector.select(job, scored_segments, job_logger)

    def _reframe_and_export(
        self,
        job: Job,
        clips: List[Clip],
        job_logger: JobLogger
    ) -> List[Clip]:
        from video.processor import VideoProcessor
        from video.reframer import AutoReframer
        from video.subtitles import SubtitleGenerator

        processor = VideoProcessor(config.get_section("export"), self.gpu.device)

        # Map user's aspect ratio selection to resolution
        shorts_cfg = config.get_section("shorts")
        aspect_text = job.config.get("reframe", "")
        res_map = {
            "9:16 Vertical (1080×1920)": (1080, 1920),
            "16:9 Landscape (1920×1080)": (1920, 1080),
            "4:3 (1440×1080)": (1440, 1080),
            "1:1 Square (1080×1080)": (1080, 1080),
            "21:9 Ultrawide (2560×1080)": (2560, 1080),
        }
        if aspect_text in res_map:
            shorts_cfg["target_resolution"] = list(res_map[aspect_text])

        reframer = AutoReframer(shorts_cfg, self.gpu.device)

        # Apply caption preset + font from job.config on top of yaml defaults
        sub_cfg = dict(config.get_section("subtitles"))
        job_cfg = job.config if isinstance(job.config, dict) else {}

        # The yaml `subtitles` section bakes in default visual fields (white text,
        # yellow highlight, black outline, bottom position, a stock font). The ASS
        # engine applies EVERY one of these as an override on top of the chosen
        # preset — which smothered each preset's own styling so every preset
        # rendered as the same white+yellow caption. Drop them here so the selected
        # preset controls appearance; only the user's EXPLICIT UI overrides (below)
        # are layered back on. This mirrors the live preview's _current_caption_style.
        for _k in ("font_color", "highlight_color", "outline_color",
                   "outline_width", "position", "font_family", "font_size"):
            sub_cfg.pop(_k, None)

        if job_cfg.get("caption_preset"):
            sub_cfg["style"] = job_cfg["caption_preset"]
        if job_cfg.get("font_family"):
            sub_cfg["font_family"] = job_cfg["font_family"]
        if job_cfg.get("caption_color"):
            sub_cfg["font_color"] = job_cfg["caption_color"]
        if job_cfg.get("highlight_color"):
            sub_cfg["highlight_color"] = job_cfg["highlight_color"]
        if job_cfg.get("caption_size", 0) > 0:
            sub_cfg["font_size"] = job_cfg["caption_size"]
        if job_cfg.get("caption_position"):
            sub_cfg["position"] = job_cfg["caption_position"]
        if "caption_y_offset" in job_cfg:
            sub_cfg["y_offset"] = job_cfg["caption_y_offset"]
        subtitle_gen = SubtitleGenerator(sub_cfg)

        final_clips = []
        for i, clip in enumerate(clips):
            if self._cancelled:
                break

            progress = 0.90 + (0.10 * (i + 1) / len(clips))
            self._update_progress(job, f"exporting_clip_{clip.id}", progress)

            if aspect_text == "None (Original)":
                reframed_path = job.output_dir / f"{job.id}_clip_{clip.id}_original.mp4"
                processor.extract_clip(
                    job.input_video, clip.start_time, clip.end_time, reframed_path, job_logger
                )
            else:
                reframed_path = reframer.reframe(
                    job.input_video,
                    clip.start_time,
                    clip.end_time,
                    job.output_dir / f"{job.id}_clip_{clip.id}_reframed.mp4",
                    job_logger
                )

            # Try subtitles; if they fail (color issues, etc.), skip them — always deliver a clean clip
            subtitle_path = None
            try:
                subtitle_path = subtitle_gen.generate(
                    clip,
                    job.output_dir / f"{job.id}_clip_{clip.id}_subtitles.srt",
                    job_logger,
                    hinglish=job_cfg.get("hinglish", False),
                )
            except Exception as e:
                job_logger.warning("Subtitle generation failed, exporting clean clip", error=str(e))

            # Check for cinematic effects config.
            effects_cfg = job.config.get("effects") if isinstance(job.config, dict) else None
            if effects_cfg:
                # Inject target resolution into effects config for filter scaling.
                effects_cfg["target_width"] = res_map.get(aspect_text, (1080, 1920))[0]
                effects_cfg["target_height"] = res_map.get(aspect_text, (1080, 1920))[1]

            # Check for border config.
            border_cfg = job_cfg.get("border") if isinstance(job.config, dict) else None

            # Check for background music config.
            music_cfg: Optional[dict] = None
            raw_music = job.config.get("music") if isinstance(job.config, dict) else None
            if isinstance(raw_music, dict) and raw_music.get("music_path"):
                music_cfg = raw_music
                job_logger.info(
                    "Background music enabled",
                    path=music_cfg["music_path"],
                    volume=music_cfg.get("volume", 0.3),
                )

            # ── Frame-level blur pre-process (OpenCV, same as preview) ──────
            blur_cfg = job_cfg.get("blur")
            if blur_cfg and blur_cfg.get("enabled", False):
                blurred_path = self._apply_blur_to_video(
                    reframed_path, blur_cfg, job_logger,
                    job.output_dir / f"{job.id}_clip_{clip.id}_blurred.mp4",
                )
                video_input = blurred_path
            else:
                video_input = reframed_path

            final_path = processor.export(
                video_input,
                subtitle_path,
                job.output_dir / f"{job.id}_clip_{clip.id}_final.mp4",
                job_logger,
                effects_config=effects_cfg,
                music_config=music_cfg,
                border_config=border_cfg,
            )

            clip.output_path = final_path
            clip.subtitle_path = subtitle_path
            final_clips.append(clip)
            self.job_manager.add_clip(job.id, clip)
            # Sync the in-memory job object so UI can read clips immediately
            if clip not in job.clips:
                job.clips.append(clip)

            # Clean up all intermediate files — only the final clip stays
            for f in [reframed_path, subtitle_path] if subtitle_path else [reframed_path]:
                if f and f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            # Also clean up .ass subtitle and blurred temp file
            ass_path = (job.output_dir / f"{job.id}_clip_{clip.id}_subtitles.ass")
            if ass_path.exists():
                try:
                    ass_path.unlink()
                except Exception:
                    pass
            blur_tmp = (job.output_dir / f"{job.id}_clip_{clip.id}_blurred.mp4")
            if blur_tmp.exists():
                try:
                    blur_tmp.unlink()
                except Exception:
                    pass

        return final_clips

    def _apply_blur_to_video(
        self,
        input_path: Path,
        blur_cfg: dict,
        job_logger: JobLogger,
        output_path: Path,
    ) -> Path:
        """Apply region blur frame-by-frame with OpenCV (same as preview widget)."""
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {input_path} for blur processing")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        intensity = max(1, int(blur_cfg.get("intensity", 15)))
        ksize = max(3, intensity // 3 * 2 + 1)
        tint_on = blur_cfg.get("tint_enabled", False)
        tint_hex = blur_cfg.get("tint_color", "#000000")
        tint_op = int(blur_cfg.get("tint_opacity", 50))
        sides = blur_cfg.get("sides", {})

        # Pre-build blur rectangles from the side config.
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
                rects.append(((w - bw) // 2, (h - bh) // 2, (w + bw) // 2, (h + bh) // 2))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for x1, y1, x2, y2 in rects:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = frame_rgb[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                if tint_on:
                    try:
                        tc = tuple(int(tint_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    except Exception:
                        tc = (0, 0, 0)
                    alpha = max(0, min(1.0, tint_op / 100.0))
                    blurred = (blurred * (1 - alpha) + np.array(tc, dtype=np.uint8) * alpha).astype(np.uint8)
                frame_rgb[y1:y2, x1:x2] = blurred
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frame_count += 1

        cap.release()
        out.release()

        # Copy audio from the original reframed video — cv2.VideoWriter writes
        # video-only (no audio stream), which would make the final export silent.
        audio_fixed = output_path.parent / f"{output_path.stem}_audiofixed{output_path.suffix}"
        try:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(output_path),        # video from OpenCV (no audio)
                    "-i", str(input_path),         # original reframed (has audio)
                    "-c:v", "copy",                # keep OpenCV-processed video
                    "-c:a", "aac",                 # encode audio
                    "-map", "0:v:0",               # video from blurred file
                    "-map", "1:a:0",               # audio from original reframed
                    "-shortest",
                    str(audio_fixed),
                ],
                capture_output=True, text=True, timeout=300,
            )
            if audio_fixed.exists():
                output_path.unlink(missing_ok=True)
                audio_fixed.rename(output_path)
                job_logger.info("Audio copied into blurred video", path=str(output_path))
        except Exception as e:
            job_logger.warning("Audio copy failed, blur export will be silent", error=str(e))

        return output_path