from __future__ import annotations

import numpy as np

from diarizer.engines.base import Segmenter
from diarizer.schemas import AudioInput, SegmentationFrames


def run_segment(audio: np.ndarray, meta: AudioInput, engine: Segmenter) -> SegmentationFrames:
    result = engine.run(audio, meta.sample_rate)
    result.audio_hash = meta.hash
    return result
