from __future__ import annotations

import os

import numpy as np

from diarizer.engines.base import Asr
from diarizer.schemas import AudioInput, WordTimestamp, WordTimestamps


class ElevenLabsScribeAsr(Asr):
    """ASR via ElevenLabs Speech-to-Text API (Scribe)."""

    def __init__(
        self,
        *,
        model: str = "scribe_v2",
        api_key_env: str = "ELEVENLABS_API_KEY",
        language: str | None = None,
        diarize: bool = False,
        tag_audio_events: bool = False,
    ):
        self._model = model
        self._api_key_env = api_key_env
        self._language = language
        self._diarize = diarize
        self._tag_audio_events = tag_audio_events

    def run(self, audio: np.ndarray, sr: int) -> WordTimestamps:
        raise NotImplementedError("ElevenLabsScribeAsr requires run_input(audio, meta) so it can stream the source file.")

    def run_input(self, audio: np.ndarray, meta: AudioInput) -> WordTimestamps:
        from elevenlabs.client import ElevenLabs

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing ElevenLabs API key. Set {self._api_key_env} in the environment before using elevenlabs_scribe."
            )

        client = ElevenLabs(api_key=api_key)
        with open(meta.path, "rb") as f:
            transcription = client.speech_to_text.convert(
                file=f,
                model_id=self._model,
                language_code=self._language,
                diarize=self._diarize,
                tag_audio_events=self._tag_audio_events,
            )

        payload = transcription if isinstance(transcription, dict) else transcription.model_dump()
        words: list[WordTimestamp] = []
        for item in payload.get("words", []):
            token = item.get("text", "")
            token_type = item.get("type", "word")
            if token_type != "word":
                continue
            words.append(
                WordTimestamp(
                    word=token.strip(),
                    start=item["start"],
                    end=item["end"],
                    confidence=item.get("confidence"),
                )
            )

        return WordTimestamps(
            words=words,
            language=payload.get("language_code", self._language or "und"),
            audio_hash="",
        )
