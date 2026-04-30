from __future__ import annotations

from typing import Any

import numpy as np

from diarizer.engines.base import Asr, Segmenter
from diarizer.engines.modal_remote import upload_audio
from diarizer.schemas import AudioInput, SegmentationFrames, SpeakerLabels, WordTimestamps


class ModalDiarizationSegmenter(Segmenter):
    """Remote speaker diarization backed by a deployed Modal function."""

    def __init__(
        self,
        *,
        app_name: str,
        function_name: str,
        volume_name: str,
        model_id: str,
        min_speakers: int = -1,
        max_speakers: int = -1,
        skip_cache: bool = False,
    ):
        self._app_name = app_name
        self._function_name = function_name
        self._volume_name = volume_name
        self._model_id = model_id
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._skip_cache = skip_cache

    def run(self, audio: np.ndarray, sr: int) -> SegmentationFrames:
        raise NotImplementedError("ModalDiarizationSegmenter does not expose frame-level segmentation.")

    def run_labels(self, audio: np.ndarray, meta: AudioInput) -> SpeakerLabels:
        import modal

        uploaded = upload_audio(meta.path, volume_name=self._volume_name)
        fn = modal.Function.from_name(self._app_name, self._function_name)
        payload = fn.remote(
            audio_id=uploaded.hash,
            model_id=self._model_id,
            min_speakers=self._min_speakers,
            max_speakers=self._max_speakers,
            skip_cache=self._skip_cache,
        )
        return SpeakerLabels.model_validate(payload)


class ModalAsr(Asr):
    """Remote ASR backed by a deployed Modal function."""

    def __init__(
        self,
        *,
        app_name: str,
        function_name: str,
        volume_name: str,
        model_id: str,
        language: str | None = None,
        skip_cache: bool = False,
    ):
        self._app_name = app_name
        self._function_name = function_name
        self._volume_name = volume_name
        self._model_id = model_id
        self._language = language
        self._skip_cache = skip_cache

    def run(self, audio: np.ndarray, sr: int) -> WordTimestamps:
        raise NotImplementedError("ModalAsr requires run_input(audio, meta) so it can upload source audio.")

    def run_input(self, audio: np.ndarray, meta: AudioInput) -> WordTimestamps:
        import modal

        uploaded = upload_audio(meta.path, volume_name=self._volume_name)
        fn = modal.Function.from_name(self._app_name, self._function_name)
        payload = fn.remote(
            audio_id=uploaded.hash,
            model_id=self._model_id,
            language=self._language,
            skip_cache=self._skip_cache,
        )
        return WordTimestamps.model_validate(payload)
