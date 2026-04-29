"""mlx-whisper engine: ASR on Apple Neural Engine (M1/M2/M3).

Uses mlx-whisper which runs entirely on Apple MLX — no PyTorch required.
"""
from __future__ import annotations

import numpy as np

from diarizer.engines.base import Asr
from diarizer.schemas import WordTimestamp, WordTimestamps


class MlxWhisperAsr(Asr):
    def __init__(self, model: str = "mlx-community/whisper-large-v3-mlx", language: str | None = None):
        self._model = model
        self._language = language

    def run(self, audio: np.ndarray, sr: int) -> WordTimestamps:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            language=self._language,
            word_timestamps=True,
        )

        words: list[WordTimestamp] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(WordTimestamp(
                    word=w["word"].strip(),
                    start=w["start"],
                    end=w["end"],
                    confidence=w.get("probability"),
                ))

        return WordTimestamps(
            words=words,
            language=result.get("language", "en"),
            audio_hash="",
        )
