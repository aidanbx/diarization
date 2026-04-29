from __future__ import annotations

from pathlib import Path

from diarizer.schemas import SpeakerLabel, SpeakerLabels

# RTTM format: SPEAKER <file_id> 1 <start> <dur> <NA> <NA> <speaker_id> <NA> <NA>


def write_rttm(labels: SpeakerLabels, path: str | Path, file_id: str = "audio") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for seg in sorted(labels.labels, key=lambda s: s.start):
            dur = seg.end - seg.start
            f.write(f"SPEAKER {file_id} 1 {seg.start:.3f} {dur:.3f} <NA> <NA> {seg.speaker_id} <NA> <NA>\n")


def read_rttm(path: str | Path, audio_hash: str = "") -> SpeakerLabels:
    path = Path(path)
    labels: list[SpeakerLabel] = []
    speaker_ids: set[str] = set()
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "SPEAKER" or len(parts) < 10:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker_id = parts[7]
            labels.append(SpeakerLabel(segment_id=i, start=start, end=start + dur, speaker_id=speaker_id))
            speaker_ids.add(speaker_id)
    return SpeakerLabels(labels=labels, num_speakers=len(speaker_ids), audio_hash=audio_hash)
