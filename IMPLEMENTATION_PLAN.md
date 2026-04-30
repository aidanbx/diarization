# Modal Diarizer Implementation Plan

## Stack

- Diarization: `BUT-FIT/diarizen-wavlm-large-s80-md-v2`
- Transcription default: `nvidia/parakeet-tdt-0.6b-v3`
- Transcription fallback: `openai/whisper-large-v3`
- Embeddings: CAM++
- Optional final refinement: Claude

## Modal architecture

- One `modal.App`
- One shared `modal.Volume` named `diarizer-cache`
- One shared GPU image for remote stages
- One local orchestrator that uploads normalized audio and calls remote stages by `audio_id`

Shared cache layout:

```text
/cache/models/
/cache/audio/
/cache/stages/{audio_hash}/
```

## Phases

1. Repo-local packaging and Modal bootstrap.
2. Shared volume + image + model prefetch.
3. Audio upload helper with idempotent hashing and normalization.
4. DiariZen diarization stage on Modal.
5. Parakeet transcription stage on Modal.
6. Local reconciliation of words to speaker segments.
7. Optional self-enrollment, overlap detection, and LLM refinement.

## Notes

- The current repo now includes phase 1 and phase 2 starter rails.
- The current `modal_app.py` is intentionally a smoke-testable bootstrap, not the finished multi-stage production pipeline yet.
- Modal auth and Hugging Face secret provisioning must happen in your authenticated local environment before remote runs succeed.
