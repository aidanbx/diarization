"""Generate a synthetic 2-speaker fixture WAV + RTTM for tests."""
import struct
import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
OUT_WAV = OUT_DIR / "2speaker_clean_30s.wav"
OUT_RTTM = OUT_DIR / "2speaker_clean_30s.rttm"

SR = 16000
DURATION = 30  # seconds

# Speaker 0: 200Hz sine; speaker 1: 300Hz sine
# Turn layout: A 0-8s, B 10-18s, A 20-28s (silence between)
turns = [
    (0.0, 8.0, "SPEAKER_00", 200),
    (10.0, 18.0, "SPEAKER_01", 300),
    (20.0, 28.0, "SPEAKER_00", 200),
]

samples = np.zeros(SR * DURATION, dtype=np.float32)
t = np.arange(SR * DURATION) / SR

for start, end, _, freq in turns:
    s, e = int(start * SR), int(end * SR)
    samples[s:e] = 0.5 * np.sin(2 * np.pi * freq * t[s:e])

OUT_DIR.mkdir(parents=True, exist_ok=True)

with wave.open(str(OUT_WAV), "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    pcm = (samples * 32767).astype(np.int16)
    wf.writeframes(pcm.tobytes())

with open(OUT_RTTM, "w") as f:
    for start, end, speaker, _ in turns:
        dur = end - start
        f.write(f"SPEAKER 2speaker_clean_30s 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

print(f"Written {OUT_WAV} and {OUT_RTTM}")
