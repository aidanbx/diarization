"""Pipeline orchestrator.

Runs all stages in order. Each stage reads its input from disk (or from the
previous stage's output) and writes its artifact to .cache/<hash>/<stage>.json.
Stages with a valid cached artifact are skipped automatically.

Usage:
    from diarizer.pipeline import Pipeline, PipelineConfig
    cfg = PipelineConfig.from_yaml("config/default.yaml")
    result = Pipeline(cfg).run("audio.m4a")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

from diarizer.io.audio import load_audio
from diarizer.io.output import write_json, write_srt, write_txt
from diarizer.io.rttm import write_rttm
from diarizer.schemas import (
    AlignedTranscript,
    AudioInput,
    SpeakerEmbeddings,
    SpeakerLabels,
    SpeechRegions,
    WordTimestamps,
)


class PipelineConfig(BaseModel):
    models_dir: str = "models/"
    cache_dir: str = ".cache/"
    output_dir: str = "output/"
    segmenter_model: str = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_model: str = "3dspeaker_speech_campplus_sv_zh-cn_16k-common_advanced.onnx"
    vad_model: str = "silero_vad.onnx"
    asr_model: str = "mlx-community/whisper-large-v3-mlx"
    language: str | None = None
    cluster_threshold: float = 0.7
    vad_threshold: float = 0.5
    num_speakers: int = -1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text())
        models_dir = raw.get("models_dir", "models/")
        return cls(
            models_dir=models_dir,
            cache_dir=raw.get("cache_dir", ".cache/"),
            output_dir=raw.get("output_dir", "output/"),
            segmenter_model=str(Path(models_dir) / raw["segmenter"]["model"] / "model.onnx"),
            embedding_model=str(Path(models_dir) / "3dspeaker_speech_campplus_sv_zh-cn_16k-common_advanced.onnx"),
            vad_model=str(Path(models_dir) / "silero_vad.onnx"),
            asr_model=raw["asr"]["model"],
            cluster_threshold=raw["cluster"]["threshold"],
            vad_threshold=raw["vad"]["threshold"],
            num_speakers=raw.get("num_speakers", -1),
        )


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._engines_loaded = False

    def _load_engines(self) -> None:
        if self._engines_loaded:
            return
        from diarizer.engines.sherpa_onnx import SherpaOnnxSegmenter, SherpaOnnxVad, SherpaOnnxEmbedder
        from diarizer.engines.mlx_whisper import MlxWhisperAsr

        cfg = self.cfg
        self._vad_engine = SherpaOnnxVad(
            model_path=cfg.vad_model,
            threshold=cfg.vad_threshold,
        )
        self._segmenter_engine = SherpaOnnxSegmenter(
            segmentation_model=cfg.segmenter_model,
            embedding_model=cfg.embedding_model,
            num_speakers=cfg.num_speakers,
            cluster_threshold=cfg.cluster_threshold,
        )
        self._embedder_engine = SherpaOnnxEmbedder(model_path=cfg.embedding_model)
        self._asr_engine = MlxWhisperAsr(model=cfg.asr_model, language=cfg.language)
        self._engines_loaded = True

    def run(self, audio_path: str | Path, skip_cache: bool = False) -> AlignedTranscript:
        audio_path = Path(audio_path)
        print(f"Loading {audio_path.name}...")
        audio, meta = load_audio(audio_path)
        cache_dir = Path(self.cfg.cache_dir) / meta.hash
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_engines()

        # VAD
        speech = self._run_or_load(
            cache_dir / "vad.json",
            lambda: self._run_vad(audio, meta),
            skip_cache,
        )

        # Segmentation (sherpa-onnx full diarization)
        raw_labels = self._run_or_load(
            cache_dir / "segment.json",
            lambda: self._run_segment(audio, meta),
            skip_cache,
        )

        # Embed
        embeddings = self._run_or_load(
            cache_dir / "embed.json",
            lambda: self._run_embed(audio, meta, raw_labels),
            skip_cache,
        )

        # Cluster
        labels = self._run_or_load(
            cache_dir / "cluster.json",
            lambda: self._run_cluster(embeddings),
            skip_cache,
        )

        # ASR
        words = self._run_or_load(
            cache_dir / "asr.json",
            lambda: self._run_asr(audio, meta),
            skip_cache,
        )

        # Align
        transcript = self._run_or_load(
            cache_dir / "align.json",
            lambda: self._run_align(words, labels),
            skip_cache,
        )

        # Write outputs
        out_dir = Path(self.cfg.output_dir) / audio_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(transcript, out_dir / "transcript.json")
        write_srt(transcript, out_dir / "transcript.srt")
        write_txt(transcript, out_dir / "transcript.txt")
        write_rttm(labels, out_dir / "diarization.rttm", file_id=audio_path.stem)

        print(f"Output written to {out_dir}/")
        return transcript

    def run_stage(self, stage: str, audio_path: str | Path) -> Any:
        """Run a single named stage, loading cached prior stages as needed."""
        audio, meta = load_audio(audio_path)
        cache_dir = Path(self.cfg.cache_dir) / meta.hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_engines()

        if stage == "vad":
            return self._run_vad(audio, meta)
        if stage == "segment":
            return self._run_segment(audio, meta)
        if stage == "embed":
            raw_labels = _load_artifact(cache_dir / "segment.json", SpeakerLabels)
            return self._run_embed(audio, meta, raw_labels)
        if stage == "cluster":
            embeddings = _load_artifact(cache_dir / "embed.json", SpeakerEmbeddings)
            return self._run_cluster(embeddings)
        if stage == "asr":
            return self._run_asr(audio, meta)
        if stage == "align":
            words = _load_artifact(cache_dir / "asr.json", WordTimestamps)
            labels = _load_artifact(cache_dir / "cluster.json", SpeakerLabels)
            return self._run_align(words, labels)
        raise ValueError(f"Unknown stage: {stage}")

    # ── stage implementations ────────────────────────────────────────────────

    def _run_vad(self, audio: np.ndarray, meta: AudioInput) -> SpeechRegions:
        from diarizer.stages.vad import run_vad
        print("  [vad] running...")
        return run_vad(audio, meta, self._vad_engine)

    def _run_segment(self, audio: np.ndarray, meta: AudioInput) -> SpeakerLabels:
        """Use sherpa-onnx's full diarization to produce initial speaker labels."""
        from diarizer.schemas import SpeakerLabel, SpeakerLabels
        print("  [segment] running sherpa-onnx diarization...")
        results = self._segmenter_engine.run_full(audio)
        labels: list[SpeakerLabel] = []
        speaker_ids: set[str] = set()
        for i, seg in enumerate(results):
            sid = f"SPEAKER_{seg.speaker:02d}"
            labels.append(SpeakerLabel(segment_id=i, start=seg.start, end=seg.end, speaker_id=sid))
            speaker_ids.add(sid)
        return SpeakerLabels(labels=labels, num_speakers=len(speaker_ids), audio_hash=meta.hash)

    def _run_embed(self, audio: np.ndarray, meta: AudioInput, segments: SpeakerLabels) -> SpeakerEmbeddings:
        from diarizer.stages.embed import run_embed
        print("  [embed] extracting embeddings...")
        return run_embed(audio, meta, segments, self._embedder_engine)

    def _run_cluster(self, embeddings: SpeakerEmbeddings) -> SpeakerLabels:
        from diarizer.stages.cluster import run_cluster
        print("  [cluster] clustering speakers...")
        return run_cluster(embeddings, threshold=self.cfg.cluster_threshold)

    def _run_asr(self, audio: np.ndarray, meta: AudioInput) -> WordTimestamps:
        from diarizer.stages.asr import run_asr
        print("  [asr] transcribing...")
        return run_asr(audio, meta, self._asr_engine)

    def _run_align(self, words: WordTimestamps, labels: SpeakerLabels) -> AlignedTranscript:
        from diarizer.stages.align import run_align
        print("  [align] aligning words to speakers...")
        return run_align(words, labels)

    # ── cache helpers ────────────────────────────────────────────────────────

    def _run_or_load(self, cache_path: Path, fn, skip_cache: bool):
        if not skip_cache and cache_path.exists():
            print(f"  [cache] {cache_path.name}")
            # Infer model type from the result of fn's return annotation isn't
            # trivial here; we store raw dicts and return them. The pipeline
            # uses typed methods, so we re-hydrate via the schema.
            return _load_raw_cached(cache_path)
        result = fn()
        _save_artifact(cache_path, result)
        return result


def _save_artifact(path: Path, obj: BaseModel) -> None:
    path.write_text(obj.model_dump_json(indent=2))


def _load_raw_cached(path: Path):
    """Load a cached artifact as raw dict (for pipeline internal use)."""
    import json
    data = json.loads(path.read_text())
    # Detect and reconstruct the right schema type from content keys
    for schema_cls in _SCHEMA_TYPES:
        try:
            return schema_cls.model_validate(data)
        except Exception:
            continue
    return data


def _load_artifact(path: Path, cls):
    data = json.loads(path.read_text())
    return cls.model_validate(data)


# ordered by specificity so the first match wins
_SCHEMA_TYPES = [
    AlignedTranscript,
    WordTimestamps,
    SpeakerLabels,
    SpeakerEmbeddings,
    SpeechRegions,
]
