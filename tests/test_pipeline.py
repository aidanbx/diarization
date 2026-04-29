"""End-to-end pipeline test using stub engines — no model files required."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

FIXTURES = Path(__file__).parent / "fixtures"
WAV = FIXTURES / "2speaker_clean_30s.wav"


def _make_stub_segmenter():
    """sherpa-onnx segmenter stub that returns two 10s speaker segments."""
    from diarizer.schemas import SpeakerLabel, SpeakerLabels

    class StubSegmenter:
        def run(self, audio, sr):
            from diarizer.schemas import SegmentationFrames, SpeakerFrame
            return SegmentationFrames(frames=[SpeakerFrame(timestamp=0.0, scores=[1.0])], frame_shift_s=0.0, audio_hash="")

        def run_full(self, audio):
            segs = []
            for i, (s, e, spk) in enumerate([(0.0, 8.0, 0), (10.0, 18.0, 1), (20.0, 28.0, 0)]):
                seg = MagicMock()
                seg.start = s
                seg.end = e
                seg.speaker = spk
                segs.append(seg)
            return segs
    return StubSegmenter()


def _make_stub_embedder():
    from diarizer.schemas import SpeakerEmbeddings, SpeakerSegment

    class StubEmbedder:
        def run(self, audio, sr, segments):
            rng = np.random.default_rng(0)
            segs = []
            for seg in segments.labels:
                # Speaker 0 → cluster A, Speaker 1 → cluster B
                base = [1.0, 0.0] if "00" in seg.speaker_id else [0.0, 1.0]
                emb = (np.array(base) + rng.normal(scale=0.01, size=2)).tolist()
                segs.append(SpeakerSegment(
                    segment_id=seg.segment_id, start=seg.start, end=seg.end, embedding=emb
                ))
            return SpeakerEmbeddings(segments=segs, audio_hash=segments.audio_hash)
    return StubEmbedder()


def _make_stub_vad():
    from diarizer.schemas import SpeechRegion, SpeechRegions

    class StubVad:
        def run(self, audio, sr):
            return SpeechRegions(regions=[SpeechRegion(start=0.0, end=30.0)], audio_hash="")
    return StubVad()


def _make_stub_asr():
    from diarizer.schemas import WordTimestamp, WordTimestamps

    class StubAsr:
        def run(self, audio, sr):
            return WordTimestamps(
                words=[
                    WordTimestamp(word="hello", start=1.0, end=1.5),
                    WordTimestamp(word="world", start=11.0, end=11.5),
                    WordTimestamp(word="again", start=21.0, end=21.5),
                ],
                language="en",
                audio_hash="",
            )
    return StubAsr()


def test_pipeline_end_to_end(tmp_path):
    from diarizer.pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(
        models_dir=str(tmp_path / "models"),
        cache_dir=str(tmp_path / ".cache"),
        output_dir=str(tmp_path / "output"),
        cluster_threshold=0.5,
    )
    pipeline = Pipeline(cfg)

    # Inject stub engines
    pipeline._vad_engine = _make_stub_vad()
    pipeline._segmenter_engine = _make_stub_segmenter()
    pipeline._embedder_engine = _make_stub_embedder()
    pipeline._asr_engine = _make_stub_asr()
    pipeline._engines_loaded = True

    transcript = pipeline.run(WAV)

    assert len(transcript.words) == 3
    assert len(transcript.turns) > 0
    assert all(w.speaker_id.startswith("SPEAKER_") for w in transcript.words)

    # Outputs written
    out_dir = tmp_path / "output" / WAV.stem
    assert (out_dir / "transcript.json").exists()
    assert (out_dir / "transcript.srt").exists()
    assert (out_dir / "transcript.txt").exists()
    assert (out_dir / "diarization.rttm").exists()


def test_pipeline_uses_cache(tmp_path):
    from diarizer.pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(
        models_dir=str(tmp_path / "models"),
        cache_dir=str(tmp_path / ".cache"),
        output_dir=str(tmp_path / "output"),
        cluster_threshold=0.5,
    )

    call_count = {"n": 0}
    original_vad = _make_stub_vad()

    class CountingVad:
        def run(self, audio, sr):
            call_count["n"] += 1
            return original_vad.run(audio, sr)

    pipeline = Pipeline(cfg)
    pipeline._vad_engine = CountingVad()
    pipeline._segmenter_engine = _make_stub_segmenter()
    pipeline._embedder_engine = _make_stub_embedder()
    pipeline._asr_engine = _make_stub_asr()
    pipeline._engines_loaded = True

    pipeline.run(WAV)
    pipeline.run(WAV)  # second run should hit cache

    assert call_count["n"] == 1
