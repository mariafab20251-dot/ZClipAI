from typing import List, Dict, Any
from core.models import TranscriptSegment, AudioFeatures, VideoFeatures, ViralScore, ScoredSegment
from utils.logging import get_logger, JobLogger
from config import config

logger = get_logger("scorer")


class ViralScorer:
    def __init__(self, scoring_config: Dict[str, Any]):
        self.weights = scoring_config.get("weights", {
            "hook_strength": 0.25,
            "emotion_level": 0.20,
            "retention_potential": 0.20,
            "speech_energy": 0.15,
            "visual_activity": 0.10,
            "uniqueness": 0.10
        })
        self.thresholds = scoring_config.get("thresholds", {})

    def score(
        self,
        transcript: List[TranscriptSegment],
        transcript_analysis: Dict[str, Any],
        audio_features: List[AudioFeatures],
        video_features: List[VideoFeatures],
        job_logger: JobLogger
    ) -> List[ScoredSegment]:
        job_logger.info("Scoring segments", count=len(transcript))

        scored_segments = []
        analyses = transcript_analysis.get("segments", [])
        global_stats = transcript_analysis.get("global_stats", {})

        for i, segment in enumerate(transcript):
            analysis = analyses[i] if i < len(analyses) else {}
            audio = audio_features[i] if i < len(audio_features) else None
            video = video_features[i] if i < len(video_features) else None

            viral_score = self._calculate_viral_score(segment, analysis, audio, video, global_stats)

            scored = ScoredSegment(
                segment=segment,
                viral_score=viral_score,
                audio_features=audio,
                video_features=video
            )
            scored_segments.append(scored)

        scored_segments.sort(key=lambda x: x.viral_score.total, reverse=True)
        job_logger.info("Scoring complete", top_score=scored_segments[0].viral_score.total if scored_segments else 0)
        return scored_segments

    def _calculate_viral_score(
        self,
        segment: TranscriptSegment,
        analysis: Dict[str, Any],
        audio: AudioFeatures,
        video: VideoFeatures,
        global_stats: Dict[str, Any]
    ) -> ViralScore:
        score = ViralScore()

        score.hook_strength = self._score_hook(analysis)
        score.emotion_level = self._score_emotion(analysis, audio, video)
        score.retention_potential = self._score_retention(segment, analysis, global_stats)
        score.speech_energy = self._score_speech_energy(audio, segment, analysis)
        score.visual_activity = self._score_visual(video)
        score.uniqueness = self._score_uniqueness(analysis, global_stats)

        self._apply_bonuses(score, analysis, audio, video)
        self._apply_penalties(score, analysis, audio, video)

        score.calculate_total(self.weights)
        return score

    def _score_hook(self, analysis: Dict[str, Any]) -> float:
        hook = analysis.get("hook_score", 0)
        curiosity = analysis.get("curiosity_score", 0)
        questions = min(analysis.get("question_count", 0) * 0.1, 0.3)
        exclamations = min(analysis.get("exclamation_count", 0) * 0.1, 0.2)
        return min(hook + curiosity + questions + exclamations, 1.0) * 100

    def _score_emotion(self, analysis: Dict[str, Any], audio: AudioFeatures, video: VideoFeatures) -> float:
        text_emotion = analysis.get("emotion_scores", {}).get("compound", 0.5)
        text_emotion = abs(text_emotion - 0.5) * 2

        audio_emotion = 0.0
        if audio:
            # Peak loudness + vocal dynamics signal emotional spikes far better than
            # mean RMS, which just tracks overall mastering/volume.
            peak = min(getattr(audio, "peak_rms", audio.rms_energy) * 12, 1.0)
            dynamics = min(getattr(audio, "energy_variance", 0.0) * 200, 1.0)
            pitch_var = min(getattr(audio, "pitch_range", 0.0) / 150, 1.0)
            audio_emotion = (peak * 0.5 + dynamics * 0.25 + pitch_var * 0.25)

        video_emotion = 0.0
        if video and video.emotions:
            max_emotion = max(video.emotions.values()) if video.emotions else 0
            video_emotion = min(max_emotion * 2, 1.0)

        return (text_emotion * 0.5 + audio_emotion * 0.3 + video_emotion * 0.2) * 100

    def _score_retention(
        self,
        segment: TranscriptSegment,
        analysis: Dict[str, Any],
        global_stats: Dict[str, Any]
    ) -> float:
        story = analysis.get("story_score", 0)
        opinion = analysis.get("opinion_score", 0)
        humor = analysis.get("humor_score", 0)

        duration = segment.end - segment.start
        duration_score = 1.0 if 15 <= duration <= 60 else 0.5

        keywords = analysis.get("keywords", [])
        keyword_density = min(len(keywords) / 10, 1.0)

        entities = analysis.get("entities", [])
        entity_score = min(len(entities) * 0.1, 0.3)

        return (story * 0.3 + opinion * 0.2 + humor * 0.2 + duration_score * 0.15 + keyword_density * 0.1 + entity_score * 0.05) * 100

    def _score_speech_energy(self, audio: AudioFeatures, segment: TranscriptSegment, analysis: Dict[str, Any]) -> float:
        if not audio:
            return 50.0

        # Blend sustained energy (mean) with the window's loudest peak so a clip
        # with one strong emphatic beat isn't averaged down to nothing.
        mean_e = min(audio.rms_energy * 20, 1.0)
        peak_e = min(getattr(audio, "peak_rms", audio.rms_energy) * 14, 1.0)
        energy = max(mean_e, peak_e)
        pace = min(analysis.get("speech_rate", 3) / 5, 1.0)

        vad_score = 1.0 if audio.voice_activity else 0.3

        return (energy * 0.5 + pace * 0.3 + vad_score * 0.2) * 100

    def _score_visual(self, video: VideoFeatures) -> float:
        if not video:
            return 30.0

        face_score = min(video.faces_detected * 0.2, 1.0)
        motion = min(video.motion_intensity * 5, 1.0)
        activity = min(video.visual_activity_score, 1.0)
        scene = 0.3 if video.scene_change else 0.0

        return (face_score * 0.4 + motion * 0.3 + activity * 0.2 + scene * 0.1) * 100

    def _score_uniqueness(self, analysis: Dict[str, Any], global_stats: Dict[str, Any]) -> float:
        keywords = analysis.get("keywords", [])
        global_keywords = set()
        for seg_analysis in global_stats.get("segment_keywords", []):
            global_keywords.update(seg_analysis)

        unique_keywords = [k for k in keywords if k not in global_keywords or keywords.count(k) == 1]
        uniqueness = min(len(unique_keywords) / max(len(keywords), 1), 1.0) if keywords else 0.3

        entities = analysis.get("entities", [])
        entity_uniqueness = min(len(entities) * 0.15, 0.5)

        return (uniqueness * 0.7 + entity_uniqueness * 0.3) * 100

    def _apply_bonuses(self, score: ViralScore, analysis: Dict, audio: AudioFeatures, video: VideoFeatures):
        if audio and audio.laughter_detected:
            score.bonuses["laughter"] = 5

        if audio and audio.applause_detected:
            score.bonuses["applause"] = 5

        if analysis.get("opinion_score", 0) > 0.5:
            score.bonuses["strong_opinion"] = 5

        if video and video.emotions:
            max_emotion = max(video.emotions.values())
            if max_emotion > 0.8:
                score.bonuses["visual_surprise"] = 5

        if analysis.get("hook_score", 0) > 0.7:
            score.bonuses["strong_hook"] = 5

    def _apply_penalties(self, score: ViralScore, analysis: Dict, audio: AudioFeatures, video: VideoFeatures):
        if audio and not audio.voice_activity:
            score.penalties["low_audio_quality"] = 10

        pauses = analysis.get("pause_density", 0)
        if pauses > 0.3:
            score.penalties["long_pauses"] = min(pauses * 10, 15)

        if analysis.get("word_count", 0) < 5:
            score.penalties["too_short"] = 15

        if analysis.get("speech_rate", 0) < 1:
            score.penalties["slow_speech"] = 5