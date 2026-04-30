from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from diarizer.schemas import AudioInput

TARGET_SR = 16000


def load_audio(path: str | Path) -> tuple[np.ndarray, AudioInput]:
    """Load any audio file → 16kHz mono float32, return (waveform, AudioInput)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    waveform = _decode_audio(path)
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


def _decode_audio(path: Path) -> np.ndarray:
    """Decode audio to 16 kHz mono float32 PCM using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
