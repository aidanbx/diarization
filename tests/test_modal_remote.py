from pathlib import Path

from diarizer.engines.modal_remote import (
    audio_id_for_file,
    mounted_audio_path,
    mounted_stage_path,
    remote_audio_path,
    remote_stage_path,
)


def test_audio_id_is_sha256(tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"hello diarizer")

    audio_id = audio_id_for_file(sample)

    assert len(audio_id) == 64
    assert all(ch in "0123456789abcdef" for ch in audio_id)


def test_remote_paths_are_stable():
    audio_id = "abc123"

    assert remote_audio_path(audio_id).as_posix() == "/audio/abc123.flac"
    assert mounted_audio_path(audio_id).as_posix() == "/cache/audio/abc123.flac"
    assert remote_stage_path(audio_id, "diarize").as_posix() == "/stages/abc123/diarize.json"
    assert mounted_stage_path(audio_id, "diarize").as_posix() == "/cache/stages/abc123/diarize.json"
