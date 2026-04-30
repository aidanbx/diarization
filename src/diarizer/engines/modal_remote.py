from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from diarizer.schemas import AudioInput

DEFAULT_VOLUME_NAME = "diarizer-cache"
CACHE_ROOT = PurePosixPath("/cache")
VOLUME_MODELS_ROOT = PurePosixPath("/models")
VOLUME_AUDIO_ROOT = PurePosixPath("/audio")
VOLUME_STAGES_ROOT = PurePosixPath("/stages")

MODELS_ROOT = CACHE_ROOT / "models"
AUDIO_ROOT = CACHE_ROOT / "audio"
STAGES_ROOT = CACHE_ROOT / "stages"


def audio_id_for_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_audio_path(audio_id: str) -> PurePosixPath:
    return VOLUME_AUDIO_ROOT / f"{audio_id}.flac"


def remote_stage_dir(audio_id: str) -> PurePosixPath:
    return VOLUME_STAGES_ROOT / audio_id


def remote_stage_path(audio_id: str, stage_name: str) -> PurePosixPath:
    return remote_stage_dir(audio_id) / f"{stage_name}.json"


def mounted_audio_path(audio_id: str) -> PurePosixPath:
    return AUDIO_ROOT / f"{audio_id}.flac"


def mounted_stage_path(audio_id: str, stage_name: str) -> PurePosixPath:
    return STAGES_ROOT / audio_id / f"{stage_name}.json"


def upload_audio(
    local_path: str | Path,
    *,
    volume_name: str = DEFAULT_VOLUME_NAME,
    force: bool = False,
) -> AudioInput:
    import modal

    src = Path(local_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    audio_id = audio_id_for_file(src)
    remote_path = remote_audio_path(audio_id)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    try:
        existing = volume.listdir(remote_path.as_posix())
    except Exception:
        existing = []
    duration_s: float
    if existing and not force:
        duration_s = _probe_duration(src)
        return AudioInput(
            path=remote_path.as_posix(),
            sample_rate=16000,
            duration_s=duration_s,
            hash=audio_id,
        )

    with tempfile.TemporaryDirectory(prefix="diarizer-modal-audio-") as tmp_dir:
        normalized = Path(tmp_dir) / f"{audio_id}.flac"
        _normalize_audio(src, normalized)
        duration_s = _probe_duration(normalized)
        with volume.batch_upload(force=force) as batch:
            batch.put_file(normalized, remote_path.as_posix())
            batch.put_file(
                io.BytesIO(
                    _json_bytes(
                    {
                        "source_path": str(src),
                        "audio_id": audio_id,
                        "sample_rate": 16000,
                        "duration_s": duration_s,
                    }
                    )
                ),
                remote_stage_path(audio_id, "audio").as_posix(),
            )

    return AudioInput(
        path=remote_path.as_posix(),
        sample_rate=16000,
        duration_s=duration_s,
        hash=audio_id,
    )


def _normalize_audio(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _json_bytes(obj: dict[str, object]) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
