# Architecture

## Stage pipeline

```
AudioInput
    │
    ├─[vad]────────► SpeechRegions
    │
    ├─[segment]────► SpeakerLabels (sherpa-onnx full diarization)
    │
    ├─[embed]──────► SpeakerEmbeddings (CAM++ per segment)
    │
    ├─[cluster]────► SpeakerLabels (agglomerative, cosine distance)
    │
    ├─[asr]────────► WordTimestamps (mlx-whisper, word-level)
    │
    └─[align]──────► AlignedTranscript (words + speaker turns)
```

## Stage I/O contracts

Each stage is a pure function: typed schema in, typed schema out. Every output is written to `.cache/<audio_hash>/<stage>.json` and reused on subsequent runs. Re-running a single stage with `diarize stage <name> <audio>` is always safe.

| Stage | Input schemas | Output schema | Engine |
|---|---|---|---|
| vad | AudioInput | SpeechRegions | SherpaOnnxVad |
| segment | AudioInput | SpeakerLabels | SherpaOnnxSegmenter (full diarization) |
| embed | AudioInput, SpeakerLabels | SpeakerEmbeddings | SherpaOnnxEmbedder |
| cluster | SpeakerEmbeddings | SpeakerLabels | scikit-learn agglomerative |
| asr | AudioInput | WordTimestamps | MlxWhisperAsr |
| align | WordTimestamps, SpeakerLabels | AlignedTranscript | pure Python |

## Engine backends

Engines are behind abstract interfaces in `engines/base.py`. The active engine is selected by `config/default.yaml → engine.*`. Stages never import engine modules directly.

| Engine file | Capabilities | Platform |
|---|---|---|
| `sherpa_onnx.py` | VAD, Segmentation, Embedding | M1/M2/M3 CoreML |
| `mlx_whisper.py` | ASR | Apple Neural Engine |

## Output formats

For each audio file processed, the pipeline writes to `output/<stem>/`:
- `transcript.json` — full AlignedTranscript schema
- `transcript.srt` — subtitle file with speaker labels
- `transcript.txt` — plain text, one turn per line
- `diarization.rttm` — industry-standard diarization format

## Caching

Cache key = first 16 hex chars of SHA-256 of the audio file content. Stored at `.cache/<hash>/`. Delete with `diarize clean <audio>` or pass `--no-cache`.
