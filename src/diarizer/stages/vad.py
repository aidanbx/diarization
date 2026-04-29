from __future__ import annotations

import numpy as np

from diarizer.engines.base import Vad
from diarizer.schemas import AudioInput, SpeechRegions


def run_vad(audio: np.ndarray, meta: AudioInput, engine: Vad) -> SpeechRegions:
    result = engine.run(audio, meta.sample_rate)
    result.audio_hash = meta.hash
    return result
