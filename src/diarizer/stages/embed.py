from __future__ import annotations

import numpy as np

from diarizer.engines.base import Embedder
from diarizer.schemas import AudioInput, SpeakerEmbeddings, SpeakerLabels


def run_embed(audio: np.ndarray, meta: AudioInput, segments: SpeakerLabels, engine: Embedder) -> SpeakerEmbeddings:
    result = engine.run(audio, meta.sample_rate, segments)
    result.audio_hash = meta.hash
    return result
