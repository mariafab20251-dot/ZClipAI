"""Candidate clip generation.

The old pipeline scored raw Whisper segments (acoustic breath-groups, ~3-15s)
and shipped them as clips. A viral short is a *narrative unit* (hook -> build ->
payoff), not a breath-group. This module rebuilds sentences from word timestamps,
then emits overlapping, sentence-aligned candidate windows in the desired duration
band. Every downstream stage (analysis, scoring, selection) then operates on these
real candidate clips instead of arbitrary ASR boundaries.

A candidate is itself a TranscriptSegment, so no downstream code needs to change:
the "segment" being scored is now a plausible clip rather than a fragment.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from core.models import TranscriptSegment, WordTimestamp
from utils.logging import get_logger, JobLogger

logger = get_logger("candidate_generator")

_TERMINALS = (".", "!", "?", "…")


@dataclass
class _Sentence:
    start: float
    end: float
    text: str
    words: List[WordTimestamp]


class CandidateGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.min_duration = float(config.get("min_duration", 15.0))
        self.max_duration = float(config.get("max_duration", 60.0))
        self.stride = float(config.get("stride", 6.0))
        self.merge_gap = float(config.get("merge_gap", 1.5))
        self.tail_pad = float(config.get("tail_pad", 0.3))

    def generate(
        self,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger,
        target_duration: float = None,
    ) -> List[TranscriptSegment]:
        if not transcript:
            return []

        # Allow the UI's requested clip length to bias the preferred window size,
        # while still respecting the configured min/max band.
        if target_duration:
            target_duration = max(self.min_duration, min(self.max_duration, float(target_duration)))

        if not self.enabled:
            job_logger.info("Candidate generation disabled, using raw segments")
            return transcript

        sentences = self._build_sentences(transcript)
        job_logger.info("Rebuilt sentences from words", sentences=len(sentences))

        candidates = self._build_windows(sentences, target_duration)
        job_logger.info(
            "Candidate clips generated",
            candidates=len(candidates),
            min_dur=self.min_duration,
            max_dur=self.max_duration,
        )
        return candidates

    # ----- sentence reconstruction -------------------------------------------------
    def _build_sentences(self, transcript: List[TranscriptSegment]) -> List[_Sentence]:
        # Flatten all words in order. Fall back to segment-level units when a
        # segment carries no word timestamps.
        words: List[WordTimestamp] = []
        for seg in transcript:
            if seg.words:
                words.extend(seg.words)
            else:
                words.append(WordTimestamp(word=seg.text, start=seg.start, end=seg.end))

        words = [w for w in words if w.end > w.start and w.word and w.word.strip()]
        words.sort(key=lambda w: w.start)
        if not words:
            return []

        sentences: List[_Sentence] = []
        cur: List[WordTimestamp] = []

        def flush():
            if not cur:
                return
            sentences.append(_Sentence(
                start=cur[0].start,
                end=cur[-1].end,
                text="".join(w.word for w in cur).strip(),
                words=list(cur),
            ))

        for i, w in enumerate(words):
            # A long pause before this word also ends the current sentence so that
            # silence gaps don't glue unrelated thoughts together.
            if cur:
                gap = w.start - cur[-1].end
                if gap > self.merge_gap:
                    flush()
                    cur = []
            cur.append(w)
            if w.word.strip().endswith(_TERMINALS):
                flush()
                cur = []
        flush()
        return sentences

    # ----- overlapping window generation -------------------------------------------
    def _build_windows(self, sentences: List[_Sentence], target_duration: float) -> List[TranscriptSegment]:
        if not sentences:
            return []

        total_span = sentences[-1].end - sentences[0].start
        # Whole input is shorter than a single clip -> one candidate covering it all.
        if total_span <= self.min_duration:
            return [self._make_candidate(0, sentences, 0, len(sentences) - 1)]

        # Targets we try to land each window's end on, per anchor.
        mid = (self.min_duration + self.max_duration) / 2
        target_lengths = sorted({self.min_duration, target_duration or mid, mid, self.max_duration})

        seen: set = set()
        candidates: List[TranscriptSegment] = []
        last_anchor_start = -1e9

        for si, sent in enumerate(sentences):
            # Thin anchors by stride to bound candidate count / overlap.
            if sent.start - last_anchor_start < self.stride:
                continue
            last_anchor_start = sent.start

            for tlen in target_lengths:
                ei = self._find_end_index(sentences, si, tlen)
                if ei is None:
                    continue
                dur = sentences[ei].end - sent.start
                if dur < self.min_duration - 1e-6 or dur > self.max_duration + 1e-6:
                    continue
                key = (round(sent.start, 2), round(sentences[ei].end, 2))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(self._make_candidate(len(candidates), sentences, si, ei))

        # Safety net: never return nothing.
        if not candidates:
            ei = self._find_end_index(sentences, 0, target_duration or mid) or len(sentences) - 1
            candidates.append(self._make_candidate(0, sentences, 0, ei))

        candidates.sort(key=lambda c: c.start)
        for i, c in enumerate(candidates):
            c.id = i
        return candidates

    def _find_end_index(self, sentences: List[_Sentence], start_idx: int, target_len: float):
        """Return the sentence index whose end lands closest to start+target_len
        without exceeding max_duration."""
        start_t = sentences[start_idx].start
        best_idx = None
        best_diff = 1e18
        for ei in range(start_idx, len(sentences)):
            dur = sentences[ei].end - start_t
            if dur > self.max_duration:
                # Allow the first overshoot only if nothing valid found yet.
                if best_idx is None and ei == start_idx:
                    return ei
                break
            diff = abs(dur - target_len)
            if diff < best_diff:
                best_diff = diff
                best_idx = ei
        return best_idx

    def _make_candidate(self, cid: int, sentences: List[_Sentence], start_idx: int, end_idx: int) -> TranscriptSegment:
        chosen = sentences[start_idx:end_idx + 1]
        words: List[WordTimestamp] = []
        for s in chosen:
            words.extend(s.words)
        text = " ".join(s.text for s in chosen).strip()
        return TranscriptSegment(
            id=cid,
            start=chosen[0].start,
            end=chosen[-1].end + self.tail_pad,
            text=text,
            words=words,
            confidence=1.0,
        )
