from pathlib import Path
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
WAV = FIXTURES / "2speaker_clean_30s.wav"
RTTM = FIXTURES / "2speaker_clean_30s.rttm"


def test_load_audio():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from diarizer.io.audio import load_audio

    waveform, meta = load_audio(WAV)
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert meta.sample_rate == 16000
    assert abs(meta.duration_s - 30.0) < 0.5
    assert len(meta.hash) == 16


def test_rttm_roundtrip(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from diarizer.io.rttm import read_rttm, write_rttm

    labels = read_rttm(RTTM, audio_hash="test")
    assert labels.num_speakers == 2
    assert len(labels.labels) == 3

    out = tmp_path / "out.rttm"
    write_rttm(labels, out)
    labels2 = read_rttm(out, audio_hash="test")

    assert len(labels2.labels) == len(labels.labels)
    for a, b in zip(labels.labels, labels2.labels):
        assert abs(a.start - b.start) < 0.001
        assert abs(a.end - b.end) < 0.001
        assert a.speaker_id == b.speaker_id


def test_output_writers(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from diarizer.schemas import AlignedTranscript, AlignedWord, SpeakerTurn
    from diarizer.io.output import write_json, write_srt, write_txt

    transcript = AlignedTranscript(
        words=[
            AlignedWord(word="hello", start=0.0, end=0.5, speaker_id="SPEAKER_00"),
            AlignedWord(word="world", start=0.6, end=1.0, speaker_id="SPEAKER_01"),
        ],
        turns=[
            SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=0.5, text="hello"),
            SpeakerTurn(speaker_id="SPEAKER_01", start=0.6, end=1.0, text="world"),
        ],
        audio_hash="test",
    )
    write_json(transcript, tmp_path / "out.json")
    write_srt(transcript, tmp_path / "out.srt")
    write_txt(transcript, tmp_path / "out.txt")

    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "out.srt").exists()
    assert (tmp_path / "out.txt").exists()
    assert "SPEAKER_00" in (tmp_path / "out.txt").read_text()
