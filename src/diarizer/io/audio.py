from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np

from diarizer.schemas import AudioInput

TARGET_SR = 16000


def load_audio(path: str | Path) -> tuple[np.ndarray, AudioInput]:
    """Load any audio file → 16kHz mono float32, return (waveform, AudioInput)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    waveform, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    duration_s = waveform.shape[0] / TARGET_SR
    file_hash = _hash_file(path)

    meta = AudioInput(
        path=str(path.resolve()),
        sample_rate=TARGET_SR,
        duration_s=duration_s,
        hash=file_hash,
    )
    return waveform.astype(np.float32), meta


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
