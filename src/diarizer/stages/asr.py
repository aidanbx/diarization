from __future__ import annotations

import numpy as np

from diarizer.engines.base import Asr
from diarizer.schemas import AudioInput, WordTimestamps


def run_asr(audio: np.ndarray, meta: AudioInput, engine: Asr) -> WordTimestamps:
    result = engine.run(audio, meta.sample_rate)
    result.audio_hash = meta.hash
    return result
