# Diarizer

Speaker diarization and transcription experiments with two execution paths:

- Local path: `sherpa-onnx` diarization + local ASR engines
- Modal path: DiariZen diarization on GPU + swappable ASR backends

## Current status

- Local chunk extraction and local pipeline are working.
- Modal-backed DiariZen + Whisper is working on the test chunks and the long-form file.
- Whisper now supports prompt and keyterm hints in the Modal path.
- Modal-backed Parakeet is wired in as a separate GPU image so it can evolve independently from the DiariZen runtime.

## Repository layout

- `src/diarizer/` core pipeline, stages, schemas, engines
- `config/` YAML presets for local and Modal runs
- `modal_app.py` Modal app definition and remote functions
- `tests/` lightweight pipeline and config tests

## Setup

Recommended environment:

```bash
conda activate diarization
python -m pip install -e ".[dev,modal,api]"
```

## Modal setup

Authenticate Modal:

```bash
modal token new
```

Set secrets:

```bash
modal secret create huggingface HUGGINGFACE_HUB_TOKEN=your_token_here
modal secret create elevenlabs ELEVENLABS_API_KEY=your_key_here
```

Deploy the Modal app:

```bash
PYTHONPATH=src modal deploy modal_app.py
```

## Config presets

Modal presets:

- `config/modal-diarizen-whisper.yaml`
- `config/modal-diarizen-parakeet.yaml`
- `config/modal-diarizen-parakeet-1.1b.yaml`
- `config/modal-diarizen-whisper-hardrefs.yaml`
- `config/modal-diarizen-elevenlabs.yaml`

Local presets:

- `config/default.yaml`
- `config/whisper-base.yaml`

## Running the pipeline

Run a chunk with the Modal Whisper backend:

```bash
PYTHONPATH=src python -m diarizer.cli run --no-cache -c config/modal-diarizen-whisper.yaml "data/26-04-28_Contact_3_chunk_01_000524.m4a"
```

Run the long file with the same backend:

```bash
PYTHONPATH=src python -m diarizer.cli run --no-cache -c config/modal-diarizen-whisper.yaml "data/26-04-28 Contact #3.m4a"
```

Run a hard-reference window with Whisper plus keyterm hints:

```bash
PYTHONPATH=src python -m diarizer.cli run --no-cache -c config/modal-diarizen-whisper-hardrefs.yaml "data/hard_sections/26-04-28_Contact_3_hard_refs_00h20m00s_to_00h28m30s.m4a"
```

Run the same window with Parakeet 1.1B:

```bash
PYTHONPATH=src python -m diarizer.cli run --no-cache -c config/modal-diarizen-parakeet-1.1b.yaml "data/hard_sections/26-04-28_Contact_3_hard_refs_00h20m00s_to_00h28m30s.m4a"
```

Extract the current hard comparison windows from the long file:

```bash
python scripts/extract_hard_sections.py "data/26-04-28 Contact #3.m4a"
```

Outputs are written under the configured `output_dir`, for example:

- `output_modal_whisper/<audio_stem>/transcript.txt`
- `output_modal_whisper/<audio_stem>/transcript.json`
- `output_modal_whisper/<audio_stem>/diarization.rttm`

## Notes on the Modal path

- Audio is normalized locally to 16 kHz mono FLAC and uploaded to the shared Modal volume.
- DiariZen returns speaker labels directly, so the local embed-and-recluster path is skipped for `segmenter_engine=modal_diarizen`.
- ASR runs on the full uploaded audio. Speaker labels are assigned afterward by timestamp alignment, so diarization is not pre-chopping the audio before transcription.
- Alignment now snaps short out-of-segment words to nearby speaker spans and repairs some tiny `SPEAKER_UNK` islands, which helps the "one person speaking but split into unknown fragments" cases.
- Remote cache keys now include model and prompt inputs so A/B tests do not accidentally reuse stale ASR results.

## Tests

```bash
python3 -m compileall modal_app.py src tests
conda run -n diarization env PYTHONPATH=src python -m pytest tests/test_pipeline.py tests/test_config.py tests/test_modal_remote.py -q
```

## Next improvements

- Finish benchmarking Whisper vs Parakeet 1.1B on the extracted hard windows
- Expand keyterm handling into a first-class CLI/config workflow
- Add a direct comparison command for multiple backend outputs
