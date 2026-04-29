# Diarization Tool — Agent Conventions

## Quick Start

```bash
conda activate diarization
python scripts/download_models.py          # idempotent — pulls all model files to models/
pytest tests/ -v                           # all tests must pass before committing
diarize run path/to/audio.m4a             # full pipeline → output/
```

## Rules

1. **Every stage is a pure function** over typed schemas. Stages take Pydantic models in, return Pydantic models out. No global state.
2. **Every stage writes its output to disk** at `.cache/<audio_hash>/<stage>.json`. The pipeline skips stages with a valid cached artifact.
3. **Engines are behind interfaces** — `stages/*.py` call `engine.run()`, never `import sherpa_onnx` directly. Engine is selected by `config.engine.*`.
4. **Tests must pass** — run `pytest tests/ -v` before any commit. Never commit broken tests.
5. **Type annotations required** on all public functions. Run `pyright` or `mypy` if unsure.

## Stage Map

| Stage | Input | Output | File | Reference |
|---|---|---|---|---|
| VAD | AudioInput | SpeechRegions | `stages/vad.py` | [Silero VAD](https://github.com/snakers4/silero-vad) |
| Segment | AudioInput + SpeechRegions | SegmentationFrames | `stages/segment.py` | [pyannote-segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| Embed | AudioInput + SpeakerLabels | SpeakerEmbeddings | `stages/embed.py` | [CAM++ / 3D-Speaker](https://arxiv.org/abs/2303.00332) |
| Cluster | SpeakerEmbeddings | SpeakerLabels | `stages/cluster.py` | agglomerative clustering |
| ASR | AudioInput | WordTimestamps | `stages/asr.py` | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) |
| Align | WordTimestamps + SpeakerLabels | AlignedTranscript | `stages/align.py` | [WhisperX](https://github.com/m-bain/whisperX) |

## Engine Backends

| Engine | File | Capabilities |
|---|---|---|
| sherpa-onnx | `engines/sherpa_onnx.py` | VAD, Segmentation, Embedding — M1 CoreML |
| mlx-whisper | `engines/mlx_whisper.py` | ASR — Apple Neural Engine |

## Key References

- **sherpa-onnx speaker diarization**: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- **sherpa-onnx model zoo**: https://github.com/k2-fsa/sherpa-onnx/releases
- **pyannote-audio pipeline source** (canonical structure): https://github.com/pyannote/pyannote-audio
- **WhisperX alignment**: https://github.com/m-bain/whisperX
- **DiariZen tutorial** (read for understanding): https://arxiv.org/abs/2604.21507
- **LLM refinement** (future stage): https://arxiv.org/abs/2509.15082
- **DER benchmarking**: https://arxiv.org/abs/2509.26177

## Out of Scope (first pass)

Self-enrollment, overlap detection, LLM refinement, cloud GPU backend, DER benchmarking scripts.
These are planned for a second pass — don't add them yet.

## Schema location

`src/diarizer/schemas.py` — all inter-stage Pydantic models live here. Add new fields there, not inline.
