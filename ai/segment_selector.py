from typing import List, Dict, Any, Optional
from core.models import ScoredSegment, Clip, TranscriptSegment
from utils.logging import get_logger, JobLogger

logger = get_logger("segment_selector")


class SegmentSelector:
    def __init__(self, selection_config: Dict[str, Any], thresholds: Dict[str, Any], target_duration: Optional[int] = None):
        self.config = selection_config
        self.thresholds = thresholds
        self.min_score = thresholds.get("min_viral_score", 40)
        self.min_duration = target_duration or thresholds.get("min_clip_duration", 15)
        self.max_duration = thresholds.get("max_clip_duration", 1200)
        self.overlap_threshold = thresholds.get("overlap_threshold", 0.3)
        self.min_gap = thresholds.get("min_gap_between_clips", 5.0)
        self.expand_boundaries = selection_config.get("expand_boundaries", True)
        self.boundary_expansion = selection_config.get("boundary_expansion", 2.0)
        self.prefer_complete_sentences = selection_config.get("prefer_complete_sentences", True)
        self.diversity_factor = selection_config.get("diversity_factor", 0.3)
        self.min_score_fallback = selection_config.get("min_score_fallback", True)

    def select(self, job, scored_segments: List[ScoredSegment], job_logger: JobLogger) -> List[Clip]:
        job_logger.info("Selecting clips", total_segments=len(scored_segments))

        candidates = [s for s in scored_segments if s.viral_score.total >= self.min_score]
        job_logger.info("Candidates above threshold", count=len(candidates))

        # FALLBACK (fixes RC: "fewer clips than requested with no explanation").
        # If nothing clears the bar, don't return empty — fall back to the best
        # available so the user always gets clips.
        if not candidates and self.min_score_fallback and scored_segments:
            job_logger.warning(
                "No candidates cleared the score threshold; falling back to top-scored",
                threshold=self.min_score,
                best_available=round(scored_segments[0].viral_score.total, 1),
            )
            candidates = list(scored_segments)

        candidates = self._adjust_boundaries(candidates)
        candidates = self._filter_by_duration(candidates)
        selected = self._select_diverse(candidates, job.num_clips)
        clips = self._create_clips(selected)

        job_logger.info("Selection complete", selected=len(clips), requested=job.num_clips)
        return clips

    def _adjust_boundaries(self, segments: List[ScoredSegment]) -> List[ScoredSegment]:
        # Candidates from the CandidateGenerator are already sentence-aligned and
        # properly sized, so we keep their boundaries verbatim. We only fall back to
        # ±expansion + sentence-snapping for raw segments (candidate gen disabled).
        for seg in segments:
            start = seg.segment.start
            end = seg.segment.end
            words = seg.segment.words

            if self.expand_boundaries:
                if words and self.prefer_complete_sentences:
                    start = self._find_sentence_start(words, start)
                    end = self._find_sentence_end(words, end)
                start = max(0, start - self.boundary_expansion)
                end = end + self.boundary_expansion

            seg.adjusted_start = start
            seg.adjusted_end = end

        return segments

    def _find_sentence_start(self, words: List, target_start: float) -> float:
        for i, word in enumerate(words):
            if word.start >= target_start:
                if i > 0 and words[i-1].word.strip().endswith(('.', '!', '?')):
                    return words[i].start
                return word.start
        return target_start

    def _find_sentence_end(self, words: List, target_end: float) -> float:
        for i in range(len(words) - 1, -1, -1):
            if words[i].end <= target_end:
                if words[i].word.strip().endswith(('.', '!', '?')):
                    return words[i].end
                if i < len(words) - 1:
                    return words[i+1].start
                return words[i].end
        return target_end

    def _filter_by_duration(self, segments: List[ScoredSegment]) -> List[ScoredSegment]:
        filtered = []
        for seg in segments:
            start = seg.adjusted_start if seg.adjusted_start is not None else seg.segment.start
            end = seg.adjusted_end if seg.adjusted_end is not None else seg.segment.end
            duration = end - start

            if duration <= self.max_duration:
                # In-band (or short): keep the real, sentence-aligned boundaries.
                # B2 FIX: do NOT re-center short clips to an exact length — that
                # discarded the sentence snapping and produced mid-sentence cuts.
                seg.adjusted_start = start
                seg.adjusted_end = end
                filtered.append(seg)
            else:
                # Genuinely too long: trim the TAIL, keeping the hook-aligned start.
                seg.adjusted_start = start
                seg.adjusted_end = start + self.max_duration
                filtered.append(seg)
        return filtered

    def _select_diverse(self, candidates: List[ScoredSegment], max_clips: int) -> List[ScoredSegment]:
        """Greedy MMR-style selection: pick the highest-scoring clip, then keep
        picking clips that are both high-scoring AND temporally far from what's
        already chosen. This is what `diversity_factor` was always supposed to do
        (it was previously dead config). Candidates must not time-overlap.
        """
        # Work on a score-sorted copy.
        pool = sorted(candidates, key=lambda c: c.viral_score.total, reverse=True)
        selected: List[ScoredSegment] = []
        used_ranges: List[tuple] = []

        if not pool:
            return selected

        # Normalizer for the diversity term: spread over the whole timeline.
        span = max((self._range(c)[1] for c in pool), default=1.0) or 1.0

        while pool and len(selected) < max_clips:
            best = None
            best_idx = -1
            best_mmr = -1e18

            for idx, cand in enumerate(pool):
                cstart, cend = self._range(cand)

                # Hard constraint: reject real time-overlap with anything selected.
                if self._overlaps(cstart, cend, used_ranges):
                    continue

                score_norm = cand.viral_score.total / 100.0
                # Distance to the NEAREST already-selected clip (0..1).
                if used_ranges:
                    nearest = min(
                        self._gap_distance(cstart, cend, us, ue) for us, ue in used_ranges
                    )
                    div = min(nearest / span, 1.0)
                else:
                    div = 1.0

                mmr = (1 - self.diversity_factor) * score_norm + self.diversity_factor * div
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = cand
                    best_idx = idx

            if best is None:
                # Everything left overlaps a selection; relax to pure non-overlap.
                break

            selected.append(best)
            used_ranges.append(self._range(best))
            pool.pop(best_idx)

        # Return in timeline order for a natural reading/preview experience.
        selected.sort(key=lambda c: self._range(c)[0])
        return selected

    def _range(self, c: ScoredSegment) -> tuple:
        start = c.adjusted_start if c.adjusted_start is not None else c.segment.start
        end = c.adjusted_end if c.adjusted_end is not None else c.segment.end
        return (start, end)

    def _overlaps(self, start: float, end: float, used_ranges: List[tuple]) -> bool:
        for us, ue in used_ranges:
            inter = min(end, ue) - max(start, us)
            if inter > 0:
                dur = max(end - start, 1e-6)
                if inter / dur > self.overlap_threshold:
                    return True
        return False

    def _gap_distance(self, start: float, end: float, us: float, ue: float) -> float:
        # 0 if the two intervals touch/overlap, else the silence gap between them.
        if end < us:
            return us - end
        if start > ue:
            return start - ue
        return 0.0

    def _create_clips(self, segments: List[ScoredSegment]) -> List[Clip]:
        clips = []
        for i, seg in enumerate(segments):
            start, end = self._range(seg)

            # Serialize WordTimestamps for subtitle generation.
            word_dicts = [
                {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                for w in seg.segment.words
            ]
            clip = Clip(
                id=i + 1,
                start_time=start,
                end_time=end,
                duration=end - start,
                viral_score=seg.viral_score.total,
                transcript=seg.segment.text,
                metadata={
                    "words": word_dicts,
                    "hook_strength": seg.viral_score.hook_strength,
                    "emotion_level": seg.viral_score.emotion_level,
                    "retention_potential": seg.viral_score.retention_potential,
                    "speech_energy": seg.viral_score.speech_energy,
                    "visual_activity": seg.viral_score.visual_activity,
                    "uniqueness": seg.viral_score.uniqueness,
                    "original_segment_id": seg.segment.id,
                    "heuristic_score": seg.heuristic_score,
                    "llm_score": seg.llm_score,
                    "llm_reason": seg.llm_reason,
                }
            )
            clips.append(clip)
        return clips