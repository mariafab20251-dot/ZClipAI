# CLIPPING ENGINE AUDIT REPORT
### AI Viral Clip Extraction Engine — Deep Architectural & Algorithmic Audit

**Audited project:** `D:\GitHub\ClippingTool`
**Audit date:** 2026-06-29
**Scope:** Full repository (≈3,400 LOC excluding the `bkkkkkk/` backup copy and Whisper model files)
**Status of this document:** Assessment & recommendations only. **No production code has been modified.**

---

## 1. Executive Summary

The tool is a well-organized, cleanly-engineered desktop pipeline (PySide6 UI → audio extract → Whisper transcription → multi-modal analysis → scoring → selection → reframe/subtitle/export). The *software engineering* is solid: clean module boundaries, dataclass models, caching, config management, graceful degradation.

**But the core promise — "find the most viral moments" — is not actually implemented.** What the engine calls "viral scoring" is a hand-tuned sum of shallow keyword counts and audio loudness, computed **on whatever arbitrary chunks Whisper happened to emit**. There is no semantic understanding, no LLM, no notion of "hook → build-up → payoff," and — most damaging — **no real candidate generation**. The unit that gets scored and shipped is the Whisper segment, which is an artifact of silence/punctuation, not a unit of meaning or virality.

### The single most important finding

> **The clips feel random because the engine never decides where a clip *should* begin or end based on content. It scores Whisper's transcription segments (typically 3–15 seconds of speech bounded by pauses), picks the top-N by a keyword-counting heuristic, then pads them by ±2 seconds. A "viral moment" is a narrative arc; a Whisper segment is a breath group. The tool is optimizing the wrong object entirely.**

Everything else (weak keyword lists, loudness≈emotion proxy, unused diversity logic, the disabled emotion model) compounds this, but **segmentation is the root cause.** Even a perfect scorer cannot produce good clips if it can only choose from bad candidate boundaries.

### Verdict

The current viral-selection strategy is **fundamentally limited and should be replaced**, not tuned. The good news: the surrounding infrastructure (transcription, audio/video feature extraction, reframing, export, caching, UI) is reusable. The fix is concentrated in three files (`segment_selector.py`, `scorer.py`, `transcript_analyzer.py`) plus a new candidate-generation stage and an LLM ranking step.

### Top 7 priorities (full roadmap in §15)

| # | Change | Impact on clip quality | Effort | Risk |
|---|--------|------------------------|--------|------|
| 1 | **Candidate generation** — merge Whisper segments into overlapping, sentence-aware windows (15–60s) instead of scoring raw segments | 🔴 Very High | Medium | Low |
| 2 | **LLM-based viral ranking** (cheap model) over candidate transcripts | 🔴 Very High | Medium | Medium |
| 3 | **Hook/payoff-aware boundary cutting** (start on hook, end on resolution) | 🟠 High | Medium | Low |
| 4 | Replace keyword-count scoring with a real signal blend (LLM + audio prosody peaks) | 🟠 High | Medium | Low |
| 5 | Fix the gap/overlap selection bug that silently drops good clips | 🟠 High | Low | Low |
| 6 | Re-enable & fix emotion/audio-peak signals; remove dead weights | 🟡 Medium | Low | Low |
| 7 | UX: show *why* a clip scored high; let users pick the niche/strategy | 🟡 Medium | Medium | Low |

---

## 2. Current Architecture

### 2.1 Module map (verified against source)

```
main.py                 → launches PySide6 app
ui/main_window.py       → input controls, progress, clip list, job history
core/
  pipeline.py           → 8-stage orchestration (the spine)
  models.py             → dataclasses: TranscriptSegment, AudioFeatures,
                          VideoFeatures, ViralScore, ScoredSegment, Clip, Job
  config.py             → pydantic settings, singleton ConfigManager, env overrides
  job_manager.py        → SQLite job persistence
  exceptions.py
ai/
  transcriber.py        → faster-whisper, word timestamps
  transcript_analyzer.py→ regex/keyword/VADER NLP per segment   ← weak link
  scorer.py             → weighted heuristic viral score          ← weak link
  segment_selector.py   → threshold + non-overlap pick + pad      ← weak link / buggy
audio/
  extractor.py          → ffmpeg → 16kHz mono wav
  analyzer.py           → librosa features, VAD, laughter/applause heuristics
video/
  analyzer.py           → InsightFace, (FER emotion - disabled), motion, scene
  reframer.py           → 9:16 auto-reframe with face tracking
  processor.py          → clip cut + export (ffmpeg)
  subtitles.py          → SRT/ASS subtitle burn-in
utils/                  → gpu, cache (json + npz), logging
config/settings.yaml    → all tunables
```

> **Note:** A complete duplicate of the project exists at `bkkkkkk/`. This is a backup/dead copy and should be removed from the working tree (it doubles grep noise and risks editing the wrong file). It is excluded from all findings below.

### 2.2 What's genuinely good

- Clean separation of concerns; each analyzer is independently testable.
- Caching (`get_or_compute`) keyed on file mtime — re-runs are cheap.
- Graceful degradation everywhere (missing spaCy, missing InsightFace, subtitle failures → still ships a clip).
- Word-level timestamps are captured (`word_timestamps=True`) — **this is the raw material a good engine needs, and it is currently underused.**
- Reframing/subtitle/export pipeline is production-grade and worth keeping as-is.

---

## 3. Processing Pipeline (stage-by-stage trace)

Traced from `core/pipeline.py::run()`.

| Stage | Input | Output | Implementation | Verdict |
|-------|-------|--------|----------------|---------|
| 1. Extract audio | video | 16kHz mono WAV | ffmpeg (`audio/extractor.py`) | ✅ Fine |
| 2. Transcribe | WAV | `List[TranscriptSegment]` w/ word ts | faster-whisper large-v3, `vad_filter=True`, `condition_on_previous_text=False`, `temperature=0` | ✅ Good config, **but segments become the clip unit — see §5** |
| 3. Analyze transcript | segments | per-segment dict of scores | regex patterns + keyword lists + VADER + spaCy | ⚠️ Shallow, English-only, keyword-bound |
| 4. Analyze audio | WAV + segments | `List[AudioFeatures]` | librosa RMS/centroid/pitch/tempo; webrtcvad; laughter/applause heuristics | ⚠️ Features OK; laughter/applause detectors are guesses |
| 5. Analyze video | video + segments | `List[VideoFeatures]` | InsightFace faces, motion, scene; **emotion disabled in yaml** | ⚠️ Emotion off; per-segment random seeks are slow |
| 6. Score | all of the above | `List[ScoredSegment]` sorted desc | weighted sum + bonuses/penalties | 🔴 Core weakness |
| 7. Select clips | scored segments | `List[Clip]` | threshold → boundary pad → duration clamp → non-overlap greedy | 🔴 Buggy + simplistic |
| 8. Reframe/export | clips | final mp4s | reframer + subtitles + processor | ✅ Solid |

### 3.1 Critical data-flow observation

Stages 3, 4, 5 all iterate **`for ... in transcript`** and produce **one feature object per Whisper segment, index-aligned**. The scorer then zips them by list index (`audio_features[i]`, `video_features[i]`). This means:

- **The clip granularity is frozen at stage 2** (transcription). Nothing downstream can propose a *better* boundary than what Whisper emitted, except a fixed ±2s pad.
- Index alignment is fragile: if any analyzer returns a different-length list (e.g., a skipped segment), the scorer silently misaligns audio/video to the wrong text with `if i < len(...)` guards that hide the bug rather than catch it.

---

## 4. Viral Detection Pipeline — How a "Viral Moment" Is Currently Decided

This is the heart of the audit. The honest answer to *"how does this tool decide what's viral?"*:

### 4.1 It is a keyword-and-loudness heuristic on breath-group chunks.

`ViralScore.total` (0–100) is a weighted sum of six components (`ai/scorer.py`):

| Component | Weight | What it *actually* measures (verified in code) |
|-----------|--------|------------------------------------------------|
| `hook_strength` | 0.25 | Regex match of first 10 words against ~5 patterns (`what/how/why…`, `secret/hidden`, `stop/wait`) + count of `?`/`!` |
| `emotion_level` | 0.20 | VADER sentiment *distance from neutral* (0.5×) + audio RMS loudness & pitch deviation (0.3×) + video emotion (0.2×, **but emotion model is disabled → always 0**) |
| `retention_potential` | 0.20 | Regex story-connective match + opinion-marker match + humor-word match + "is duration 15–60s?" + keyword count |
| `speech_energy` | 0.15 | Audio RMS loudness + words-per-second + VAD on/off |
| `visual_activity` | 0.10 | Face count + motion + Laplacian variance + scene-change flag |
| `uniqueness` | 0.10 | Count of keywords not seen in other segments |

Plus flat bonuses (laughter +5, applause +5, strong opinion +5, hook>0.7 +5, visual surprise +5) and penalties (no-VAD −10, long pauses, <5 words −15, slow speech −5).

### 4.2 What this means in practice

- **"Hook" = does the sentence start with a question word.** A genuinely gripping cold-open statement ("I lost everything in 30 seconds") scores **zero** hook because it doesn't start with what/how/why and has no `?`.
- **"Emotion" ≈ loudness + presence of words like "amazing."** A quiet, devastating confession scores low. A loud boring rant scores high.
- **"Retention potential" rewards the word "because" and being 15–60s long.** It has no concept of whether the segment actually *resolves* anything.
- **"Uniqueness" rewards rare nouns**, which correlates with *names and jargon*, not with novelty of idea.

None of these measure the things that actually drive virality (§6). The score is essentially **"loud sentence containing trigger words, of medium length."**

### 4.3 The decision procedure (`segment_selector.py`)

1. Keep segments with `total >= 40`.
2. Pad each by ±2s, snap to sentence boundaries via word punctuation.
3. Clamp duration to [target, 90]; if too short, **re-center on the segment midpoint and force it to exactly `min_duration`** (this throws away the sentence-snapping just done).
4. Greedy walk down the score-sorted list, skipping anything that overlaps >30% or is within 5s of an already-picked clip.
5. Take first N.

---

## 5. Root Cause Analysis — Why Clips Feel Random

Ranked by contribution to the "random clips" problem.

### 🔴 RC-1 — The clip unit is the Whisper segment, not a narrative unit *(the dominant cause)*

A viral short is **a complete micro-story**: setup → tension → payoff, or hook → curiosity gap → resolution. Whisper segments are **acoustic breath groups** split on ~500ms silences and punctuation — usually 3–15s of a single sentence or two. Consequences:

- A clip starts mid-thought and ends before the punchline, because the punchline was in the *next* Whisper segment.
- The "best" 8-second segment in isolation is often a fragment that's meaningless without its lead-in.
- Boundary "expansion" (±2s) is a band-aid that cannot reconstruct a 40-second story from a 7-second fragment.

**This alone is sufficient to make output feel random**, regardless of how good the scorer is. *You are choosing the best item from a list of the wrong items.*

### 🔴 RC-2 — No semantic understanding of content

The scorer cannot tell the difference between:
- "So anyway, I went to the store" (filler), and
- "And that's when I realized my business partner had been stealing from me for years" (a payoff).

Both are ~medium length, both may be at similar volume, neither contains a regex hook word. To a keyword counter they're interchangeable. **Virality lives in meaning, and meaning is exactly what this pipeline never models.**

### 🟠 RC-3 — Loudness is used as a proxy for emotion and importance

`rms_energy * 10/20` dominates `emotion_level` and `speech_energy` (35% of total weight combined). This systematically rewards shouting, music stings, and audio compression artifacts over genuine emotional or intellectual peaks. Quiet but riveting moments are penalized.

### 🟠 RC-4 — Selection logic bug drops good clips silently

In `_select_non_overlapping` (`segment_selector.py:119-123`), the gap check is:
```python
if abs(start - used_end) < self.min_gap or abs(end - used_start) < self.min_gap:
    gap_ok = False
```
`abs()` makes this reject a candidate whose *start is within 5s of an already-used clip's end* — i.e., two genuinely good back-to-back moments can't both be selected, **and the rejection is order-dependent on score**. Combined with the overlap test, on short videos this frequently yields *fewer clips than requested* with no explanation, and the clips you do get are scattered to satisfy the gap constraint rather than chosen for quality. This makes results feel arbitrary across runs.

### 🟠 RC-5 — The "force to min_duration" re-centering destroys boundaries

`_filter_by_duration` re-centers short candidates on their midpoint and sets an exact-length window (`segment_selector.py:85-89`), discarding the sentence-aware start/end found moments earlier. The clip then starts/ends mid-sentence anyway — the most visible symptom of "random" cuts.

### 🟡 RC-6 — Disabled/placebo signals inflate confidence without adding signal

- `emotion_detection.enabled: false` in yaml → `video.emotions` is always `{}` → the 20% video contribution to emotion and the "visual_surprise" bonus are **dead code that never fires**. The architecture *claims* facial-emotion awareness it doesn't have.
- `laughter`/`applause` detectors are crude variance thresholds with no validation; they fire on any energetic noisy passage.
- `diversity_factor: 0.3` is defined in config and **never read anywhere** — there is no actual diversity enforcement, so multiple near-identical clips can be selected.

### 🟡 RC-7 — Everything is English- and keyword-locked

Hook/emotion/story/opinion/humor are all hardcoded English word lists. Any other language, or any phrasing that doesn't use the exact trigger words, scores ~0 on the text components. The system is brittle to vocabulary it wasn't hand-fed.

---

## 6. What Actually Makes a Clip Viral (and whether the tool measures it)

| Viral driver | Measured today? | How it *should* be detected |
|--------------|-----------------|-----------------------------|
| Strong hook in first 1–3s | ❌ Only "starts with question word" | LLM: "does the opening create a curiosity gap / stakes / pattern interrupt?" |
| Curiosity gap / open loop | ❌ | LLM detects unresolved question that the clip later answers |
| Emotional intensity (any valence) | ⚠️ loudness + sentiment-distance | LLM emotion classification + audio prosody *peaks* (not mean loudness) |
| Conflict / tension | ❌ | LLM: detects disagreement, stakes, confrontation |
| Surprise / plot twist | ❌ | LLM: detects reversal vs. prior context |
| Story payoff / resolution | ❌ | LLM: does the segment *complete* a thought it opened? |
| Self-contained (no missing context) | ❌ | LLM: "would this make sense to someone who didn't see the lead-up?" |
| Quotable / memorable line | ❌ | LLM: extract a punchy standalone line |
| High information density / value | ❌ | LLM: "does the viewer learn something usable?" |
| Controversy / strong stance | ⚠️ opinion-marker words | LLM: detects a defensible/divisive claim |
| Humor | ⚠️ "lol/haha" words | LLM (audio laughter as secondary signal) |
| Pacing / energy | ⚠️ WPM + loudness | prosody variance, scene-change rate (secondary) |

**Conclusion:** Of ~12 real viral drivers, the tool meaningfully measures **none**; it has weak proxies for ~4. The qualities that separate a viral clip from a random one are precisely the ones requiring semantic comprehension — which is why a keyword engine cannot succeed here.

---

## 7. Critique of the Current Approach (first-principles)

> *If I were building a viral clip engine today, would I use this approach?* **No.**

The architecture implicitly assumes virality is a **bag-of-features property of a fixed text chunk**. It is not. Virality is a **semantic, structural property of a self-contained narrative whose boundaries you get to choose.** Two foundational mistakes follow:

1. **Boundaries are an output, not an input.** A good engine *generates* candidate boundaries (where could a great clip start and end?) and evaluates many overlapping options. This engine treats Whisper's boundaries as given and only nudges them ±2s.
2. **Shallow features cannot represent meaning.** Regex and keyword counts are a 2015-era approach. With a cheap LLM now costing fractions of a cent per clip-candidate, hand-maintaining English trigger-word lists is both worse *and* more work.

The heuristic stack is also **uncalibrated**: weights (0.25/0.20/…) and the `min_viral_score=40` threshold are guesses with no ground-truth tuning. The 0–100 "score" looks precise but is not comparable across videos (uniqueness is relative to the video; loudness is relative to mastering).

---

## 8. Code Quality Review

### Bugs / correctness

| ID | File:line | Severity | Issue |
|----|-----------|----------|-------|
| B1 | `segment_selector.py:119-123` | High | `abs()` gap check rejects valid non-adjacent clips; order-dependent; yields fewer-than-requested clips (RC-4) |
| B2 | `segment_selector.py:85-94` | High | Re-centering to exact duration discards sentence-aware boundaries (RC-5) |
| B3 | `scorer.py:36-38` | Medium | Index-aligned zip of transcript/audio/video lists; silent misalignment if any length differs |
| B4 | `config.py:266` | Medium | `clip_duration` is sourced from `min_clip_duration` (15), **not** the UI's duration spinner default (30) — `to_app_config` is inconsistent with the actual job path |
| B5 | `scorer.py:80-81` | Low | `question_count`/`exclamation_count` come from `text.count("?")`, but Whisper segment text rarely contains `?`/`!` with `temperature=0` and no punctuation guarantee → hook nearly always under-counts |
| B6 | `transcript_analyzer.py:100` | Low | Same `?`/`!` counting on lowercased text; fine, but redundant with hook |
| B7 | `video/analyzer.py:108` | Medium (perf) | `cap.set(POS_FRAMES)` random-seek per sampled frame per segment — very slow on long videos / AV1 |
| B8 | `audio/analyzer.py:129-146` | Low | Laughter/applause thresholds are unvalidated magic numbers; high false-positive rate feeds +5 bonuses |

### Dead / unused code & config

- `selection.diversity_factor` — defined, never used.
- `video.emotion_detection.enabled: false` — entire emotion path is dead; `_score_emotion`'s video term and `visual_surprise` bonus can never fire.
- `ARCHITECTURE.md` lists modules that don't exist (`audio/voice_activity.py`, `video/face_tracker.py`, `video/exporter.py`, `ui/worker.py`, `ui/preview.py`, `utils/helpers.py`) — doc drift.
- `bkkkkkk/` — full duplicate tree; delete.

### Maintainability

- Scoring constants are scattered as magic numbers inside methods (`* 10`, `* 20`, `* 0.2`, `min(..., 0.3)`), not in config. Tuning requires code edits.
- No tests despite `tests/` and the architecture doc's "golden master tests for scoring." `tests/__init__.py` is empty.
- `_calculate_pause_density` computes `total_gap` but never uses it.

---

## 9. Performance Review

| Area | Current | Issue | Recommendation |
|------|---------|-------|----------------|
| Video analysis | per-segment random seeks (`POS_FRAMES`) | Seeks are the dominant cost on long/AV1 video; O(segments × samples) seeks | Single linear decode pass; sample at fixed FPS; aggregate to segments after |
| Audio analysis | full-file librosa load + piptrack | `piptrack` is expensive and only a global mean pitch is used | Use `pyin`/`yin` or drop to per-window pitch; compute once (already global) ✓ |
| Whisper | large-v3, beam_size=1 (yaml) vs 5 (config default) | Inconsistent; large-v3 is slow | Consider `distil-large-v3` (already downloaded) for 2× speed at ~same WER for clip selection |
| Scoring/NLP | spaCy per segment | Fine for typical lengths | Batch with `nlp.pipe()` if needed |
| Parallelism | analyzers run sequentially | audio & video analysis are independent | Run audio/video/transcript-analysis concurrently (3 threads) |
| Export | per-clip ffmpeg, cleans intermediates | Good | Keep; optionally NVENC |

Performance is **not** the project's problem — quality is. But the video random-seek (B7) is worth fixing because it dominates wall-clock and discourages running on full-length source.

---

## 10. AI Prompt Review

**There are no LLM prompts in the project.** Verified by full-repo grep: no `openai`, `anthropic`, `gemini`, `llm`, `completion`, or API-key usage anywhere outside Whisper's tokenizer files. "AI" currently means Whisper + VADER + spaCy + regex.

This is the **biggest opportunity**: introducing a single well-designed LLM ranking prompt over candidate transcripts will do more for clip quality than any amount of heuristic tuning. Prompt design is specified in §11.3 and §12.

---

## 11. Alternative Architecture Proposal

### 11.1 Design principle

> **Generate many candidate clips with content-aware boundaries → score them with a cheap LLM that understands virality → calibrate with objective audio/visual signals → select a diverse top-N.**

This is the *candidate-generation + reranking* pattern used by every serious clipping product (Opus Clip, Vizard, etc.). It directly fixes RC-1, RC-2, RC-3.

### 11.2 Proposed pipeline

```
Transcribe (keep — word timestamps)
        │
        ▼
[NEW] Semantic chunking          → merge Whisper segments into sentences,
                                    then into topic/utterance units
        │
        ▼
[NEW] Candidate generation       → sliding, overlapping windows over sentence
                                    boundaries: every plausible 15–60s clip that
                                    starts at a sentence start & ends at a
                                    sentence end (hundreds of candidates)
        │
        ▼
[REUSE] Objective signals        → audio prosody PEAKS, scene changes, faces,
                                    laughter — computed once, mapped onto any window
        │
        ▼
[NEW] LLM ranking (cheap model)  → batch candidates; each gets viral scores +
                                    reason + best hook line + self-contained? flag
        │
        ▼
[NEW] Fusion + calibration       → final = f(LLM score, prosody peak, completeness)
        │
        ▼
[REWRITE] Selection              → MMR-style diverse top-N, correct gap logic,
                                    boundaries snapped to hook start / payoff end
        │
        ▼
[REUSE] Reframe / subtitle / export (unchanged)
```

### 11.3 Why an LLM, and which one

A cheap model (e.g. **Haiku 4.5** / a small fast tier) can read a 60-second transcript candidate and answer, reliably and in JSON, the questions in §6 that regex cannot. At hundreds of candidates per video, batching keeps cost to a **few cents per video** (see §13). Premium models are **not** justified for per-candidate scoring; reserve a premium model (Opus/Sonnet tier) only for an optional final "pick the 3 best and write titles" pass where nuance pays off.

### 11.4 Hybrid scoring (recommended fusion)

```
final_score = 0.60 * llm_viral_score          # semantic: hook, payoff, surprise, value
            + 0.20 * prosody_peak_score        # objective: emotional/energy peak within window
            + 0.10 * completeness_score        # LLM flag: self-contained, clean start/end
            + 0.10 * audiovisual_bonus         # laughter/applause/scene/face — capped
```
Keep objective signals as **calibration and tie-breakers**, not primary drivers. This preserves the good feature-extraction work while letting semantics lead.

### 11.5 Boundary selection (fixes the "random cut" feel)

For each selected candidate, use word timestamps to:
1. **Start** on the first word of the hook sentence the LLM identified (not ±2s padding).
2. **End** on the last word of the resolving sentence (let the payoff land, optional +0.3s breath).
3. Snap to the nearest silence ≥150ms to avoid clipping speech.

This is the difference between a clip that "starts in the middle of a word" and one that feels intentionally edited.

---

## 12. Recommended LLM Prompt (starting point)

System role + per-candidate JSON output, batched. Modular and reusable across niches:

```
SYSTEM:
You are an expert short-form video editor who has produced thousands of viral
clips for TikTok, Reels, and Shorts. You judge whether a transcript excerpt would
make someone stop scrolling, watch to the end, and share.

For each candidate, score 0-10 on each axis and return strict JSON:
- hook:        Does the first sentence create curiosity, stakes, or a pattern interrupt?
- payoff:      Does the excerpt resolve/deliver on what it opens? Is there a punchline,
               twist, lesson, or emotional landing?
- self_contained: Would this make full sense to someone who didn't see what came before?
- emotion:     Intensity of emotion (any kind: funny, shocking, moving, infuriating).
- shareability: Would a viewer send this to a friend?
- best_hook_line: the single strongest line to open the clip (verbatim).
- ideal_start_sentence / ideal_end_sentence: where the clip should begin/end.
- reason: one sentence explaining the score.
- overall: your single 0-100 viral score.

Penalize: mid-thought starts, missing context, rambling, no payoff, generic filler.
Do not reward mere loudness or keywords. Reward genuine narrative completeness.
```

Notes:
- Output strict JSON → validate & retry on parse failure.
- Pass a short **video-level context** (title/topic) once, so "self_contained" and "surprise" are judged against the whole.
- Make the rubric **niche-configurable** (podcast, courtroom, gaming, education) by swapping the system preamble — mirrors a presets pattern, low effort, high payoff.

---

## 13. AI Cost Optimization Plan

| Task | Model tier | Rationale |
|------|-----------|-----------|
| Transcription | local Whisper (`distil-large-v3` already on disk) | No API cost; distil = ~2× faster |
| Per-candidate viral scoring | **cheapest capable LLM** (Haiku-class) | Bulk, structured, repetitive — exactly what cheap models are for. Batch 5–10 candidates/call |
| Optional final title/caption + top-3 reranking | mid/premium tier, ≤1 call/video | Nuance & marketing copy benefit from a stronger model; tiny volume |
| Code indexing / dedup / docs (this kind of audit work) | cheap model | No premium needed |

**Cost controls to build in:**
1. **Pre-filter candidates before the LLM** with the cheap objective signals (drop windows with no speech, no prosody variance, <8 words) — cut LLM volume 50–70%.
2. **Cache LLM verdicts** keyed on candidate-transcript hash (reuse the existing `get_or_compute`).
3. **Batch** multiple candidates per request.
4. Cap candidates per video (e.g., top 60 by cheap pre-score) → LLM cost becomes a few cents/video.
5. Make the LLM **optional/offline-degradable**: if no API key, fall back to the improved heuristic so the tool still runs.

Premium models are justified **only** where nuance changes the outcome at low volume (final title writing, optional top-pick reranking). Everything high-volume stays on the cheap tier.

---

## 14. Product & UX Review

| Issue | Recommendation | Effort |
|-------|----------------|--------|
| Score shown as a bare number + stars; user can't tell *why* a clip was picked | Show the LLM's `reason`, `best_hook_line`, and per-axis bars (hook/payoff/emotion). Builds trust, exposes bad picks | Low |
| No control over *what kind* of virality to target | Add a "Content type / strategy" dropdown (Podcast, Education, Funny, Drama, Courtroom) that swaps the LLM rubric | Low |
| `min_viral_score=40` is invisible and arbitrary; users get "fewer clips than asked" with no reason | Make threshold a slider; if N can't be met, say why ("only 4 segments cleared the bar") | Low |
| No preview/playback in-app before export | Embed a player or thumbnail strip for each candidate | Medium |
| Can't re-rank or reject a candidate and regenerate | Add a candidate review tab: approve/reject/adjust boundaries, then export only approved | Medium |
| Duration spinner vs. config mismatch (B4) | Single source of truth for clip duration | Low |
| No feedback loop | Let users ⭐ clips that performed well; store to tune fusion weights over time | Medium |

---

## 15. Prioritized Improvement Roadmap

Each item: **Impact / Effort / Risk / Maintainability**.

### Phase 0 — Hygiene & quick wins (1 day)
1. Delete `bkkkkkk/`; fix `ARCHITECTURE.md` drift. *(Low/Low/Low/+)*
2. Fix gap-check bug B1 and the re-center bug B2 in `segment_selector.py`. **Immediate, visible reduction in "random cut" feel.** *(High/Low/Low/+)*
3. Fix B4 duration source; remove dead `diversity_factor` or implement it. *(Med/Low/Low/+)*
4. Move scoring magic numbers into `settings.yaml`. *(Med/Low/Low/++)*

### Phase 1 — Candidate generation (the highest-leverage change) (2–4 days)
5. New `ai/candidate_generator.py`: merge Whisper segments → sentences → overlapping 15–60s windows snapped to sentence boundaries. Score *these*, not raw segments. **Fixes RC-1.** *(Very High/Medium/Low/+)*
6. Map existing audio/video features onto arbitrary windows (interval overlap), and switch audio "emotion/energy" from **mean** to **peak/variance within window**. **Fixes RC-3.** *(High/Medium/Low/+)*

### Phase 2 — LLM ranking (the quality unlock) (3–5 days)
7. New `ai/llm_ranker.py` with the §12 prompt, batching, JSON validation, caching, and **graceful offline fallback** to the improved heuristic. **Fixes RC-2.** *(Very High/Medium/Medium/+)*
8. Fusion scorer (§11.4) replacing the current weighted sum; keep heuristic as fallback path. *(High/Medium/Low/+)*
9. Hook-start / payoff-end boundary cutting from word timestamps (§11.5). **Fixes RC-5 properly.** *(High/Medium/Low/+)*

### Phase 3 — Selection & diversity (1–2 days)
10. Rewrite selection as MMR (relevance − redundancy) for genuine diversity; correct gap/overlap math; honor "N clips" or explain shortfall. *(High/Low/Low/+)*

### Phase 4 — Signals & perf polish (2–3 days)
11. Re-enable emotion (or remove its dead weight honestly); validate laughter/applause or downgrade to weak tie-breakers (B8). *(Medium/Low/Low/+)*
12. Single-pass linear video decode to kill random-seek cost (B7). *(Medium-perf/Medium/Low/+)*
13. Run audio/video/text analysis concurrently. *(Low/Low/Low/+)*

### Phase 5 — Product & feedback (ongoing)
14. UX: reasons, per-axis bars, strategy dropdown, candidate review/approve, in-app preview (§14). *(Medium/Medium/Low/+)*
15. Optional premium final pass: titles/captions + top-3 rerank. *(Medium/Low/Low/+)*
16. Add the tests promised in the architecture doc (golden-master scoring, selection unit tests). *(Med/Med/Low/++)*

### Expected outcome
- **Phase 0 alone** removes the most jarring "cut in the middle of a sentence / fewer clips than asked" symptoms.
- **Phases 1–2 are the actual fix** for "clips feel random" — moving from breath-group keyword scoring to narrative-aware, semantically-ranked candidates is the step change.
- Phases 3–5 turn a working engine into a trustworthy product.

---

## 16. Answering the Brief's Core Question

> *"Is the existing viral clip selection strategy fundamentally sound?"*

**No.** It is well-built software around an unsound core idea: that virality can be read off fixed transcription chunks with keyword counts and loudness. The two non-negotiable changes are **(1) generate content-aware candidate clips instead of scoring Whisper segments**, and **(2) judge them with a model that understands meaning.** Without these, no amount of weight-tuning will stop the clips from feeling random — because the engine is choosing the best option from a set of fundamentally wrong options.

The surrounding engineering (transcription, feature extraction, reframing, subtitles, export, caching, UI) is genuinely good and should be **kept and reused**. The intelligence layer (`scorer.py`, `segment_selector.py`, `transcript_analyzer.py`) should be **replaced** with the candidate-generation + LLM-ranking + fusion architecture in §11.

**Recommended first step:** Phase 0 + Phase 1 (candidate generation). It is low-risk, reuses all existing features, and will produce the single biggest visible jump in clip quality — before any LLM cost is incurred. Await approval before I begin implementation.
