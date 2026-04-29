"""sherpa-onnx engine: VAD, Segmentation, and Speaker Embedding on M1/CoreML.

Model files are expected at config.models_dir (default: models/).
Run scripts/download_models.py first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from diarizer.engines.base import Embedder, Segmenter, Vad
from diarizer.schemas import (
    SegmentationFrames,
    SpeakerEmbeddings,
    SpeakerFrame,
    SpeakerLabel,
    SpeakerLabels,
    SpeakerSegment,
    SpeechRegion,
    SpeechRegions,
)


class SherpaOnnxVad(Vad):
    """Stub VAD — sherpa-onnx OfflineSpeakerDiarization handles VAD internally.

    This returns the full audio as a single speech region. The real VAD pass
    happens inside SherpaOnnxSegmenter.run_full(). A standalone Silero VAD
    could be wired here in future if pre-filtering long silences is needed.
    """

    def run(self, audio: np.ndarray, sr: int) -> SpeechRegions:
        duration = len(audio) / sr
        return SpeechRegions(
            regions=[SpeechRegion(start=0.0, end=duration)],
            audio_hash="",
        )


class SherpaOnnxSegmenter(Segmenter):
    """Wraps the sherpa-onnx offline speaker diarization pipeline.

    sherpa-onnx bundles pyannote-segmentation-3.0 ONNX + CAM++ embeddings in a
    single OfflineSpeakerDiarization object. run_full() returns the final
    speaker segments; clustering is done internally by sherpa-onnx.
    """

    def __init__(
        self,
        segmentation_model: str | Path,
        embedding_model: str | Path,
        num_speakers: int = -1,  # -1 = auto
        cluster_threshold: float = 0.5,
    ):
        diar_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation_model),
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding_model),
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers,
                threshold=cluster_threshold,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(diar_config)

    def run(self, audio: np.ndarray, sr: int) -> SegmentationFrames:
        # sherpa-onnx doesn't expose raw frame scores directly; we produce a
        # synthetic single-frame placeholder so the pipeline schema is satisfied.
        # The actual segmentation result is retrieved via run_full() in the pipeline.
        frames = [SpeakerFrame(timestamp=0.0, scores=[1.0])]
        return SegmentationFrames(frames=frames, frame_shift_s=0.0, audio_hash="")

    def run_full(self, audio: np.ndarray) -> sherpa_onnx.OfflineSpeakerDiarizationResult:
        """Run the full sherpa-onnx diarization and return the result object."""
        return self._diarizer.process(audio)


class SherpaOnnxEmbedder(Embedder):
    """Extracts CAM++ embeddings for arbitrary audio segments."""

    def __init__(self, model_path: str | Path):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path))
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def run(self, audio: np.ndarray, sr: int, segments: SpeakerLabels) -> SpeakerEmbeddings:
        result: list[SpeakerSegment] = []
        for seg in segments.labels:
            s = int(seg.start * sr)
            e = int(seg.end * sr)
            chunk = audio[s:e]
            if len(chunk) < 160:  # too short to embed
                continue
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=sr, waveform=chunk)
            stream.input_finished()
            raw = self._extractor.compute(stream)
            embedding = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            result.append(SpeakerSegment(
                segment_id=seg.segment_id,
                start=seg.start,
                end=seg.end,
                embedding=embedding,
            ))
        return SpeakerEmbeddings(segments=result, audio_hash=segments.audio_hash)
