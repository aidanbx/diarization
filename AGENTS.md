# AGENTS

## Current repo status

- Local pipeline exists under `src/diarizer/`.
- `modal_app.py` is the bootstrap entrypoint for Modal work.
- `src/diarizer/engines/modal_remote.py` owns local hashing, normalization, and volume upload helpers.
- Shared cache contract:
  - `/cache/models`
  - `/cache/audio`
  - `/cache/stages/{audio_id}`

## Immediate implementation target

Use Modal as the remote execution layer for the next pipeline revision:

- DiariZen for diarization
- Parakeet for ASR
- Whisper large-v3 as fallback
- CAM++ for speaker enrollment/rescoring

Do not introduce extra orchestration systems. Modal is the orchestrator.

## Modal references

- Overview: https://modal.com/docs
- Images guide: https://modal.com/docs/guide/images
- Volumes guide: https://modal.com/docs/guide/volumes
- `modal.Volume` reference: https://modal.com/docs/reference/modal.Volume
- `modal.Image` reference: https://modal.com/docs/reference/modal.Image
- `modal.App` and entrypoints: https://modal.com/docs/guide/apps
- Secrets guide: https://modal.com/docs/guide/secrets
- `modal.Secret` reference: https://modal.com/docs/reference/modal.Secret

## DiariZen references

- Repo: https://github.com/BUTSpeechFIT/DiariZen
- Model usage:
  - `DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")`
- Installation notes from upstream:
  - PyTorch 2.1.1
  - CUDA 12.1
  - `pip install -r requirements.txt && pip install -e .`
  - install bundled `pyannote-audio`

## Parakeet references

- Model card: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NeMo install recommendation: `pip install -U nemo_toolkit['asr']`
- Inference entrypoint:
  - `nemo.collections.asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")`

## Whisper fallback reference

- Model card: https://huggingface.co/openai/whisper-large-v3
- Recommended runtime backend: faster-whisper

## Constraints and implementation notes

- Keep local and remote responsibilities separate.
- Cache everything by input-audio SHA-256.
- Prefer pure JSON artifacts for stage outputs.
- Preserve the current local pipeline while building the Modal path in parallel.
- Treat Modal dependencies as project-local via `pyproject.toml`, not as an assumed global install.
