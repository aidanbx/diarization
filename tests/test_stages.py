"""Stage unit tests using synthetic data — no model files required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diarizer.schemas import (
    AudioInput,
    SpeakerEmbeddings,
    SpeakerLabel,
    SpeakerLabels,
    SpeakerSegment,
    SpeechRegion,
    SpeechRegions,
    WordTimestamp,
    WordTimestamps,
)


# ── cluster ─────────────────────────────────────────────────────────────────

def test_cluster_two_speakers():
    from diarizer.stages.cluster import run_cluster

    rng = np.random.default_rng(42)
    # Two clearly distinct embedding clusters
    emb_a = rng.normal(loc=[1.0, 0.0], scale=0.01, size=(5, 2))
    emb_b = rng.normal(loc=[0.0, 1.0], scale=0.01, size=(5, 2))

    segments = []
    for i, e in enumerate(np.vstack([emb_a, emb_b])):
        segments.append(SpeakerSegment(segment_id=i, start=i * 2.0, end=i * 2.0 + 1.5, embedding=e.tolist()))

    embeddings = SpeakerEmbeddings(segments=segments, audio_hash="test")
    labels = run_cluster(embeddings, threshold=0.5)

    assert labels.num_speakers == 2
    assert len(labels.labels) == 10
    # all labels are well-formed
    for lbl in labels.labels:
        assert lbl.speaker_id.startswith("SPEAKER_")


def test_cluster_single_segment():
    from diarizer.stages.cluster import run_cluster

    seg = SpeakerSegment(segment_id=0, start=0.0, end=5.0, embedding=[0.5, 0.5])
    embeddings = SpeakerEmbeddings(segments=[seg], audio_hash="test")
    labels = run_cluster(embeddings)
    assert labels.num_speakers == 1
    assert len(labels.labels) == 1


def test_cluster_empty():
    from diarizer.stages.cluster import run_cluster

    embeddings = SpeakerEmbeddings(segments=[], audio_hash="test")
    labels = run_cluster(embeddings)
    assert labels.num_speakers == 0
    assert labels.labels == []


# ── align ────────────────────────────────────────────────────────────────────

def _make_labels() -> SpeakerLabels:
    return SpeakerLabels(
        labels=[
            SpeakerLabel(segment_id=0, start=0.0, end=5.0, speaker_id="SPEAKER_00"),
            SpeakerLabel(segment_id=1, start=5.5, end=10.0, speaker_id="SPEAKER_01"),
        ],
        num_speakers=2,
        audio_hash="test",
    )


def _make_words() -> WordTimestamps:
    return WordTimestamps(
        words=[
            WordTimestamp(word="hello", start=0.1, end=0.5),
            WordTimestamp(word="there", start=0.6, end=1.0),
            WordTimestamp(word="world", start=5.6, end=6.0),
        ],
        language="en",
        audio_hash="test",
    )


def test_align_basic():
    from diarizer.stages.align import run_align

    transcript = run_align(_make_words(), _make_labels())
    assert len(transcript.words) == 3
    assert transcript.words[0].speaker_id == "SPEAKER_00"
    assert transcript.words[1].speaker_id == "SPEAKER_00"
    assert transcript.words[2].speaker_id == "SPEAKER_01"


def test_align_turns():
    from diarizer.stages.align import run_align

    transcript = run_align(_make_words(), _make_labels())
    assert len(transcript.turns) == 2
    assert transcript.turns[0].speaker_id == "SPEAKER_00"
    assert "hello" in transcript.turns[0].text
    assert transcript.turns[1].speaker_id == "SPEAKER_01"


def test_align_empty_words():
    from diarizer.stages.align import run_align

    words = WordTimestamps(words=[], language="en", audio_hash="test")
    transcript = run_align(words, _make_labels())
    assert transcript.words == []
    assert transcript.turns == []
