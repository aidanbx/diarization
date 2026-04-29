from __future__ import annotations

import json
from pathlib import Path

from diarizer.schemas import AlignedTranscript, SpeakerLabels


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj.model_dump(), f, indent=2)


def write_srt(transcript: AlignedTranscript, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, turn in enumerate(transcript.turns, 1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_time(turn.start)} --> {_fmt_time(turn.end)}\n")
            f.write(f"[{turn.speaker_id}] {turn.text}\n\n")


def write_txt(transcript: AlignedTranscript, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for turn in transcript.turns:
            f.write(f"[{turn.speaker_id} {_fmt_time(turn.start)}] {turn.text}\n")


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
