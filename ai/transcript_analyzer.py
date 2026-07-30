from typing import List, Dict, Any
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from core.models import TranscriptSegment
from utils.logging import get_logger, JobLogger

logger = get_logger("transcript_analyzer")


class TranscriptAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.nlp = None
        self._load_spacy()

        self.hook_patterns = [
            r"^(what|how|why|when|where|who)\b",
            r"\b(imagine|picture this|what if)\b",
            r"\b(secret|hidden|unknown|nobody knows)\b",
            r"\b(stop|wait|don't|never)\b",
            r"\b(this will|you won't believe|mind blown)\b",
        ]

        self.curiosity_triggers = [
            "secret", "hidden", "revealed", "truth", "actually",
            "surprising", "shocking", "unexpected", "weird", "strange"
        ]

        self.emotion_keywords = {
            "excitement": ["amazing", "incredible", "fantastic", "awesome", "wow", "omg", "unbelievable"],
            "anger": ["furious", "outraged", "disgusted", "unacceptable", "ridiculous", "angry"],
            "fear": ["scary", "terrifying", "dangerous", "warning", "be careful", "afraid"],
            "joy": ["happy", "delighted", "thrilled", "wonderful", "best", "love"],
            "sadness": ["sad", "heartbreaking", "devastating", "tragic", "sorry"],
            "surprise": ["suddenly", "unexpected", "shocked", "stunned", "speechless"],
            "disgust": ["gross", "disgusting", "nasty", "revolting", "horrible"]
        }

        self.story_patterns = [
            r"\b(once|first|then|next|finally|eventually)\b",
            r"\b(because|so|therefore|consequently)\b",
            r"\b(but|however|although|despite)\b",
        ]

        self.opinion_markers = [
            "i think", "i believe", "in my opinion", "honestly",
            "frankly", "personally", "i'm convinced", "no doubt"
        ]

        self.humor_indicators = [
            "lol", "haha", "funny", "hilarious", "joke", "kidding",
            "sarcasm", "ironic", "lmao", "rofl"
        ]

    def _load_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic analysis")
            self.nlp = None

    def analyze(self, transcript: List[TranscriptSegment], job_logger: JobLogger) -> Dict[str, Any]:
        job_logger.info("Analyzing transcript", segments=len(transcript))

        results = {
            "segments": [],
            "global_stats": {}
        }

        all_text = " ".join(s.text for s in transcript)

        for segment in transcript:
            analysis = self._analyze_segment(segment, all_text)
            results["segments"].append(analysis)

        results["global_stats"] = self._compute_global_stats(transcript, results["segments"])
        job_logger.info("Transcript analysis complete")

        return results

    def _analyze_segment(self, segment: TranscriptSegment, full_text: str) -> Dict[str, Any]:
        text = segment.text.lower()
        words = text.split()

        analysis = {
            "segment_id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "duration": segment.end - segment.start,
            "word_count": len(words),
            "hook_score": self._calculate_hook_score(text),
            "curiosity_score": self._calculate_curiosity_score(text),
            "emotion_scores": self._calculate_emotion_scores(text),
            "sentiment": self._get_sentiment(text),
            "story_score": self._calculate_story_score(text),
            "opinion_score": self._calculate_opinion_score(text),
            "humor_score": self._calculate_humor_score(text),
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
            "pause_density": self._calculate_pause_density(segment),
            "speech_rate": len(words) / max(segment.end - segment.start, 0.1),
            "keywords": self._extract_keywords(text),
            "entities": self._extract_entities(segment.text),
        }

        return analysis

    def _calculate_hook_score(self, text: str) -> float:
        score = 0.0
        first_words = " ".join(text.split()[:10])

        for pattern in self.hook_patterns:
            if re.search(pattern, first_words, re.IGNORECASE):
                score += 0.3

        if text.startswith(("what", "how", "why", "when", "who")):
            score += 0.2

        return min(score, 1.0)

    def _calculate_curiosity_score(self, text: str) -> float:
        score = 0.0
        for trigger in self.curiosity_triggers:
            if trigger in text:
                score += 0.15
        return min(score, 1.0)

    def _calculate_emotion_scores(self, text: str) -> Dict[str, float]:
        scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            scores[emotion] = min(count * 0.2, 1.0)

        vader = self.sentiment_analyzer.polarity_scores(text)
        scores["compound"] = (vader["compound"] + 1) / 2
        scores["positive"] = vader["pos"]
        scores["negative"] = vader["neg"]
        scores["neutral"] = vader["neu"]

        return scores

    def _get_sentiment(self, text: str) -> Dict[str, float]:
        vader = self.sentiment_analyzer.polarity_scores(text)
        blob = TextBlob(text)
        return {
            "vader_compound": vader["compound"],
            "vader_pos": vader["pos"],
            "vader_neg": vader["neg"],
            "vader_neu": vader["neu"],
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }

    def _calculate_story_score(self, text: str) -> float:
        score = 0.0
        for pattern in self.story_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.15
        return min(score, 1.0)

    def _calculate_opinion_score(self, text: str) -> float:
        score = 0.0
        for marker in self.opinion_markers:
            if marker in text:
                score += 0.2
        return min(score, 1.0)

    def _calculate_humor_score(self, text: str) -> float:
        score = 0.0
        for indicator in self.humor_indicators:
            if indicator in text:
                score += 0.2
        return min(score, 1.0)

    def _calculate_pause_density(self, segment: TranscriptSegment) -> float:
        if not segment.words or len(segment.words) < 2:
            return 0.0

        pauses = 0
        total_gap = 0.0
        for i in range(1, len(segment.words)):
            gap = segment.words[i].start - segment.words[i-1].end
            if gap > 0.5:
                pauses += 1
                total_gap += gap

        return pauses / len(segment.words) if segment.words else 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        if not self.nlp:
            return []

        doc = self.nlp(text)
        keywords = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not token.is_stop and len(token.text) > 2:
                keywords.append(token.lemma_.lower())
        return list(set(keywords))[:10]

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        if not self.nlp:
            return []

        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def _compute_global_stats(self, transcript: List[TranscriptSegment], analyses: List[Dict]) -> Dict[str, Any]:
        if not analyses:
            return {}

        durations = [a["duration"] for a in analyses]
        hook_scores = [a["hook_score"] for a in analyses]
        emotion_compounds = [a["emotion_scores"].get("compound", 0.5) for a in analyses]

        return {
            "total_duration": sum(durations),
            "total_segments": len(transcript),
            "avg_segment_duration": sum(durations) / len(durations),
            "avg_hook_score": sum(hook_scores) / len(hook_scores),
            "avg_emotion_intensity": sum(emotion_compounds) / len(emotion_compounds),
            "peak_emotion_segment": max(analyses, key=lambda x: x["emotion_scores"].get("compound", 0.5))["segment_id"],
            "peak_hook_segment": max(analyses, key=lambda x: x["hook_score"])["segment_id"],
        }