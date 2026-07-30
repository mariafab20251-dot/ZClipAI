import librosa
import numpy as np
import webrtcvad
from typing import List, Dict, Any
from pathlib import Path
from core.models import TranscriptSegment, AudioFeatures
from utils.logging import get_logger, JobLogger

logger = get_logger("audio_analyzer")


class AudioAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_mfcc = config.get("n_mfcc", 13)
        self.hop_length = config.get("hop_length", 512)
        self.n_fft = config.get("n_fft", 2048)
        self.vad_aggressiveness = config.get("vad_aggressiveness", 3)
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

    def analyze(
        self,
        audio_path: Path,
        transcript: List[TranscriptSegment],
        job_logger: JobLogger
    ) -> List[AudioFeatures]:
        job_logger.info("Analyzing audio", audio_path=str(audio_path))

        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        duration = len(y) / sr

        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)

        times = librosa.frames_to_time(np.arange(len(rms)), hop_length=self.hop_length, sr=sr)

        # Per-frame pitch timeline (the old code computed ONE global mean pitch and
        # assigned it identically to every segment, so pitch could never
        # discriminate between segments). f0 over time lets each window get its own
        # pitch level and intonation range.
        try:
            f0 = librosa.yin(y, fmin=65, fmax=600, sr=sr, hop_length=self.hop_length)
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            f0 = np.zeros_like(rms)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo.item() if isinstance(tempo, np.ndarray) else tempo)

        features = []
        for i, segment in enumerate(transcript):
            seg_features = self._extract_segment_features(
                segment, y, sr, times, rms, spectral_centroid,
                spectral_rolloff, zcr, mfccs, f0, tempo
            )
            features.append(seg_features)

        job_logger.info("Audio analysis complete", segments=len(features))
        return features

    def _extract_segment_features(
        self,
        segment: TranscriptSegment,
        y: np.ndarray,
        sr: int,
        times: np.ndarray,
        rms: np.ndarray,
        spectral_centroid: np.ndarray,
        spectral_rolloff: np.ndarray,
        zcr: np.ndarray,
        mfccs: np.ndarray,
        f0: np.ndarray,
        tempo: float
    ) -> AudioFeatures:
        start_idx = np.searchsorted(times, segment.start, side="left")
        end_idx = np.searchsorted(times, segment.end, side="right")
        start_idx = max(0, min(start_idx, len(rms) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(rms)))

        seg_rms = rms[start_idx:end_idx]
        seg_centroid = spectral_centroid[start_idx:end_idx]
        seg_rolloff = spectral_rolloff[start_idx:end_idx]
        seg_zcr = zcr[start_idx:end_idx]
        seg_mfccs = mfccs[:, start_idx:end_idx]
        seg_f0 = f0[start_idx:end_idx]
        voiced = seg_f0[seg_f0 > 0]

        voice_activity = self._detect_voice_activity(y, sr, segment.start, segment.end)
        laughter = self._detect_laughter(seg_rms, seg_zcr, seg_centroid)
        applause = self._detect_applause(seg_rms, seg_centroid, seg_rolloff)

        # 95th-percentile peak captures the loudest emotional spike in the window
        # without being thrown off by a single clipped sample.
        peak_rms = float(np.percentile(seg_rms, 95)) if len(seg_rms) else 0.0
        pitch_mean = float(np.mean(voiced)) if len(voiced) else 0.0
        pitch_range = float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if len(voiced) > 1 else 0.0

        return AudioFeatures(
            segment_start=segment.start,
            segment_end=segment.end,
            rms_energy=float(np.mean(seg_rms)),
            spectral_centroid=float(np.mean(seg_centroid)),
            spectral_rolloff=float(np.mean(seg_rolloff)),
            zero_crossing_rate=float(np.mean(seg_zcr)),
            mfccs=[float(np.mean(seg_mfccs[i])) for i in range(self.n_mfcc)],
            pitch=pitch_mean,
            tempo=float(tempo),
            voice_activity=voice_activity,
            laughter_detected=laughter,
            applause_detected=applause,
            peak_rms=peak_rms,
            energy_variance=float(np.var(seg_rms)) if len(seg_rms) else 0.0,
            pitch_range=pitch_range
        )

    def _detect_voice_activity(self, y: np.ndarray, sr: int, start: float, end: float) -> bool:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = y[start_sample:end_sample]

        if len(segment_audio) < 320:
            return False

        segment_audio = (segment_audio * 32767).astype(np.int16)
        frame_duration = 30
        frame_size = int(sr * frame_duration / 1000)

        speech_frames = 0
        total_frames = 0

        for i in range(0, len(segment_audio) - frame_size, frame_size):
            frame = segment_audio[i:i + frame_size].tobytes()
            if len(frame) == frame_size * 2:
                try:
                    if self.vad.is_speech(frame, sr):
                        speech_frames += 1
                    total_frames += 1
                except:
                    pass

        return (speech_frames / max(total_frames, 1)) > 0.3

    def _detect_laughter(self, rms: np.ndarray, zcr: np.ndarray, centroid: np.ndarray) -> bool:
        if len(rms) < 5:
            return False

        energy_var = np.var(rms)
        zcr_mean = np.mean(zcr)
        centroid_var = np.var(centroid)

        return energy_var > 0.01 and zcr_mean > 0.1 and centroid_var > 1000

    def _detect_applause(self, rms: np.ndarray, centroid: np.ndarray, rolloff: np.ndarray) -> bool:
        if len(rms) < 10:
            return False

        energy_sustained = np.mean(rms) > 0.05
        high_freq_content = np.mean(centroid) > 3000 and np.mean(rolloff) > 5000
        energy_fluctuation = np.std(rms) / (np.mean(rms) + 1e-6) > 0.5

        return energy_sustained and high_freq_content and energy_fluctuation