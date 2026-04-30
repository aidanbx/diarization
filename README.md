# Diarizer

Speaker diarization and transcription experiments with two execution paths:

- Local path: `sherpa-onnx` diarization + local ASR engines
- Modal path: DiariZen diarization on GPU + swappable ASR backends

## Current status

- Local chunk extraction and local pipeline are working.
- Modal-backed DiariZen + Whisper is working on the test chunks.
- Modal-backed Parakeet and ElevenLabs Scribe are wired in as selectable backends, but only the Whisper path has been exercised end to end so far.

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

Outputs are written under the configured `output_dir`, for example:

- `output_modal_whisper/<audio_stem>/transcript.txt`
- `output_modal_whisper/<audio_stem>/transcript.json`
- `output_modal_whisper/<audio_stem>/diarization.rttm`

## Notes on the Modal path

- Audio is normalized locally to 16 kHz mono FLAC and uploaded to the shared Modal volume.
- DiariZen returns speaker labels directly, so the local embed-and-recluster path is skipped for `segmenter_engine=modal_diarizen`.
- Whisper output quality is already much better than the earlier local `whisper-base` path, but alignment still produces some short `SPEAKER_UNK` fragments.

## Tests

```bash
python3 -m compileall modal_app.py src tests
conda run -n diarization env PYTHONPATH=src python -m pytest tests/test_pipeline.py tests/test_config.py tests/test_modal_remote.py -q
```

## Next improvements

- Improve `align.py` to reduce `SPEAKER_UNK` fragments and over-segmentation
- Benchmark Modal Whisper vs Modal Parakeet on the same chunks
- Add a direct comparison command for multiple backend outputs
