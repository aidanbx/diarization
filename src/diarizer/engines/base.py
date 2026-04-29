from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from diarizer.schemas import (
    SpeakerEmbeddings,
    SpeakerLabels,
    SpeechRegions,
    SegmentationFrames,
    WordTimestamps,
)


class Vad(ABC):
    @abstractmethod
    def run(self, audio: np.ndarray, sr: int) -> SpeechRegions: ...


class Segmenter(ABC):
    @abstractmethod
    def run(self, audio: np.ndarray, sr: int) -> SegmentationFrames: ...


class Embedder(ABC):
    @abstractmethod
    def run(self, audio: np.ndarray, sr: int, segments: SpeakerLabels) -> SpeakerEmbeddings: ...


class Asr(ABC):
    @abstractmethod
    def run(self, audio: np.ndarray, sr: int) -> WordTimestamps: ...


class EngineBundle:
    """Container for all active engine instances, selected by config."""
    vad: Vad
    segmenter: Segmenter
    embedder: Embedder
    asr: Asr

    def __init__(self, vad: Vad, segmenter: Segmenter, embedder: Embedder, asr: Asr):
        self.vad = vad
        self.segmenter = segmenter
        self.embedder = embedder
        self.asr = asr
