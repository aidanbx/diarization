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
    def __init__(self, model_path: str | Path, threshold: float = 0.5,
                 min_speech_ms: int = 250, min_silence_ms: int = 100, sr: int = 16000):
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(model_path),
                threshold=threshold,
                min_silence_duration=min_silence_ms / 1000.0,
                min_speech_duration=min_speech_ms / 1000.0,
            ),
            sample_rate=sr,
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def run(self, audio: np.ndarray, sr: int) -> SpeechRegions:
        self._vad.accept_waveform(audio)
        self._vad.flush()
        regions: list[SpeechRegion] = []
        while not self._vad.empty():
            seg = self._vad.front
            regions.append(SpeechRegion(start=seg.start, end=seg.start + seg.duration))
            self._vad.pop()
        return SpeechRegions(regions=regions, audio_hash="")


class SherpaOnnxSegmenter(Segmenter):
    """Wraps the sherpa-onnx offline speaker diarization pipeline for segmentation.

    sherpa-onnx bundles pyannote-segmentation-3.0 ONNX + CAM++ embeddings in a
    single OfflineSpeakerDiarization object. We use it here only for the
    segmentation frame scores; clustering is done separately in stages/cluster.py.
    """

    def __init__(
        self,
        segmentation_model: str | Path,
        embedding_model: str | Path,
        num_speakers: int = -1,  # -1 = auto
        cluster_threshold: float = 0.5,
    ):
        seg_config = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(segmentation_model),
        )
        emb_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model),
        )
        cluster_config = sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=cluster_threshold,
        )
        diar_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=seg_config,
            embedding=emb_config,
            clustering=cluster_config,
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

    def run_full(self, audio: np.ndarray) -> list[sherpa_onnx.OfflineSpeakerDiarizationResult]:
        """Run the full sherpa-onnx diarization and return raw results."""
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
            embedding = self._extractor.compute(stream).tolist()
            result.append(SpeakerSegment(
                segment_id=seg.segment_id,
                start=seg.start,
                end=seg.end,
                embedding=embedding,
            ))
        return SpeakerEmbeddings(segments=result, audio_hash=segments.audio_hash)
