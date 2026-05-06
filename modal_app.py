from __future__ import annotations

import json
import os
import hashlib
import subprocess
from pathlib import Path

import modal

from diarizer.engines.modal_remote import (
    AUDIO_ROOT,
    CACHE_ROOT,
    DEFAULT_VOLUME_NAME,
    MODELS_ROOT,
    mounted_audio_path,
    mounted_stage_path,
    remote_stage_path,
    upload_audio,
)
from diarizer.schemas import SpeakerLabel, SpeakerLabels, WordTimestamp, WordTimestamps

APP_NAME = "diarizer"
HF_SECRET_NAME = "huggingface"
HF_TOKEN_ENV = "HUGGINGFACE_HUB_TOKEN"
ELEVENLABS_SECRET_NAME = "elevenlabs"

DIARIZEN_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
WHISPER_FALLBACK_MODEL_ID = "openai/whisper-large-v3"
FASTER_WHISPER_MODEL_ID = "large-v3"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
elevenlabs_secret = modal.Secret.from_name(ELEVENLABS_SECRET_NAME, environment_name=None)


def _download_hf_models() -> list[str]:
    from huggingface_hub import snapshot_download

    token = os.environ.get(HF_TOKEN_ENV)
    root = Path(MODELS_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    for model_id in [
        DIARIZEN_MODEL_ID,
        PARAKEET_MODEL_ID,
        WHISPER_FALLBACK_MODEL_ID,
    ]:
        target = root / model_id
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            token=token,
            resume_download=True,
        )
        downloaded.append(str(target))
    volume.commit()
    return downloaded


def _read_cached_json(audio_id: str, stage_name: str) -> dict | None:
    cache_path = Path(mounted_stage_path(audio_id, stage_name))
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return None


def _write_cached_json(audio_id: str, stage_name: str, payload: dict) -> None:
    cache_path = Path(mounted_stage_path(audio_id, stage_name))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2))
    volume.commit()


def _audio_path(audio_id: str) -> Path:
    return Path(mounted_audio_path(audio_id))


def _cache_stage_name(base: str, **kwargs: object) -> str:
    fingerprint = hashlib.sha256(
        json.dumps(kwargs, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{base}_{fingerprint}"


def _normalize_speaker_id(raw: object) -> str:
    text = str(raw)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return f"SPEAKER_{int(digits):02d}"
    return text.upper().replace("SPEAKER_", "SPEAKER_").replace("SPEAKER", "SPEAKER_")


base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "git")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["modal"])
)

image = (
    base_image
    .pip_install(
        "ctranslate2>=4.5,<5",
        "faster-whisper>=1.1,<2",
        "nemo_toolkit[asr]>=2.0.0,<3",
        "numpy<2",
        "sentencepiece>=0.2,<0.3",
        "torch==2.1.2",
        "torchaudio==2.1.2",
    )
    .run_commands(
        "git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git /tmp/diarizen-src",
        "python -m pip install -e '/tmp/diarizen-src/pyannote-audio[dev,testing]'",
        "python -m pip install -e /tmp/diarizen-src",
    )
    .add_local_python_source("diarizer")
)

parakeet_image = (
    base_image
    .pip_install(
        "nemo_toolkit[asr]>=2.0.0,<3",
        "sentencepiece>=0.2,<0.3",
        "torch==2.4.1",
        "torchaudio==2.4.1",
    )
    .add_local_python_source("diarizer")
)


@app.function(image=image, volumes={CACHE_ROOT.as_posix(): volume}, secrets=[hf_secret], timeout=60 * 60)
def download_models() -> list[str]:
    """Prefetch Hugging Face model snapshots into the shared Modal volume."""
    return _download_hf_models()


@app.function(image=image, gpu="A10G", timeout=10 * 60)
def gpu_check() -> str:
    """Return the visible GPU name from inside a Modal container."""
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    return out[0] if out else "no-gpu-detected"


@app.function(image=image, volumes={CACHE_ROOT.as_posix(): volume}, timeout=10 * 60)
def list_cached_files(prefix: str = "/cache") -> list[str]:
    """List cached files currently visible from the mounted Modal volume."""
    root = Path(prefix)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file())


@app.function(image=image, volumes={CACHE_ROOT.as_posix(): volume}, secrets=[hf_secret], gpu="A10G", timeout=60 * 60)
def run_diarization(
    audio_id: str,
    model_id: str = DIARIZEN_MODEL_ID,
    min_speakers: int = -1,
    max_speakers: int = -1,
    skip_cache: bool = False,
) -> dict:
    from diarizen.pipelines.inference import DiariZenPipeline

    volume.reload()
    stage_name = _cache_stage_name(
        "diarize",
        model_id=model_id,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    cached = None if skip_cache else _read_cached_json(audio_id, stage_name)
    if cached is not None:
        return cached

    audio_path = _audio_path(audio_id)
    pipeline = DiariZenPipeline.from_pretrained(model_id)
    result = pipeline(str(audio_path))

    labels: list[SpeakerLabel] = []
    speakers: set[str] = set()
    for idx, (turn, _, speaker) in enumerate(result.itertracks(yield_label=True)):
        sid = _normalize_speaker_id(speaker)
        speakers.add(sid)
        labels.append(SpeakerLabel(segment_id=idx, start=turn.start, end=turn.end, speaker_id=sid))

    payload = SpeakerLabels(labels=labels, num_speakers=len(speakers), audio_hash=audio_id).model_dump()
    _write_cached_json(audio_id, stage_name, payload)
    return payload


@app.function(image=parakeet_image, volumes={CACHE_ROOT.as_posix(): volume}, secrets=[hf_secret], gpu="A10G", timeout=60 * 60)
def run_parakeet_asr(
    audio_id: str,
    model_id: str = PARAKEET_MODEL_ID,
    language: str | None = None,
    initial_prompt: str | None = None,
    hotwords: list[str] | None = None,
    skip_cache: bool = False,
) -> dict:
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    volume.reload()
    stage_name = _cache_stage_name(
        "asr_parakeet",
        model_id=model_id,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords or [],
    )
    cached = None if skip_cache else _read_cached_json(audio_id, stage_name)
    if cached is not None:
        return cached

    audio_path = _audio_path(audio_id)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.word_seperator = " "
        decoding_cfg.segment_seperators = [".", "?", "!"]
    model.change_decoding_strategy(decoding_cfg)
    output = model.transcribe([str(audio_path)], return_hypotheses=True)
    if isinstance(output, tuple):
        output = output[0]
    output = output[0]

    words = _extract_parakeet_words(output, model=model)
    payload = WordTimestamps(words=words, language=language or getattr(output, "language", None) or "und", audio_hash=audio_id).model_dump()
    _write_cached_json(audio_id, stage_name, payload)
    return payload


@app.function(image=parakeet_image, volumes={CACHE_ROOT.as_posix(): volume}, secrets=[hf_secret], gpu="A10G", timeout=60 * 60)
def debug_parakeet_asr(
    audio_id: str,
    model_id: str = "nvidia/parakeet-tdt-1.1b",
    language: str | None = None,
) -> dict:
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    volume.reload()
    audio_path = _audio_path(audio_id)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.word_seperator = " "
        decoding_cfg.segment_seperators = [".", "?", "!"]
    model.change_decoding_strategy(decoding_cfg)
    output = model.transcribe([str(audio_path)], return_hypotheses=True)
    raw = output
    if isinstance(output, tuple):
        output = output[0]
    first = output[0]

    payload = first if isinstance(first, dict) else getattr(first, "__dict__", {})
    return {
        "raw_type": str(type(raw)),
        "output_type": str(type(output)),
        "first_type": str(type(first)),
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "has_timestamp": hasattr(first, "timestamp"),
        "has_timestamps": hasattr(first, "timestamps"),
        "has_timestep": hasattr(first, "timestep"),
        "timestamp_type": str(type(getattr(first, "timestamp", None))),
        "timestep_type": str(type(getattr(first, "timestep", None))),
        "timestamp_repr": repr(getattr(first, "timestamp", None))[:4000],
        "timestep_repr": repr(getattr(first, "timestep", None))[:4000],
        "text": getattr(first, "text", None),
        "payload_repr": repr(payload)[:4000],
    }


@app.function(image=image, volumes={CACHE_ROOT.as_posix(): volume}, secrets=[hf_secret], gpu="A10G", timeout=60 * 60)
def run_whisper_asr(
    audio_id: str,
    model_id: str = FASTER_WHISPER_MODEL_ID,
    language: str | None = None,
    initial_prompt: str | None = None,
    hotwords: list[str] | None = None,
    skip_cache: bool = False,
) -> dict:
    from faster_whisper import WhisperModel

    volume.reload()
    stage_name = _cache_stage_name(
        "asr_whisper",
        model_id=model_id,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords or [],
    )
    cached = None if skip_cache else _read_cached_json(audio_id, stage_name)
    if cached is not None:
        return cached

    audio_path = _audio_path(audio_id)
    model = WhisperModel(model_id, device="cuda", compute_type="float16")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        hotwords=", ".join(hotwords) if hotwords else None,
    )

    words: list[WordTimestamp] = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            token = (word.word or "").strip()
            if not token:
                continue
            words.append(
                WordTimestamp(
                    word=token,
                    start=word.start,
                    end=word.end,
                    confidence=getattr(word, "probability", None),
                )
            )

    payload = WordTimestamps(words=words, language=language or info.language or "und", audio_hash=audio_id).model_dump()
    _write_cached_json(audio_id, stage_name, payload)
    return payload


def _extract_parakeet_words(output, *, model=None) -> list[WordTimestamp]:
    payload = output if isinstance(output, dict) else getattr(output, "__dict__", {})
    timestamps = None
    if isinstance(payload, dict):
        timestamps = payload.get("timestamp") or payload.get("timestamps") or payload.get("word_timestamps")
    if timestamps is None:
        timestamps = getattr(output, "timestamp", None) or getattr(output, "timestamps", None) or getattr(output, "word_timestamps", None)

    words: list[WordTimestamp] = []
    if isinstance(timestamps, dict):
        for token, start, end in zip(
            timestamps.get("word", []),
            timestamps.get("start", []),
            timestamps.get("end", []),
        ):
            words.append(WordTimestamp(word=str(token).strip(), start=float(start), end=float(end)))
        return words

    if isinstance(timestamps, list):
        for item in timestamps:
            if isinstance(item, dict):
                token = item.get("word") or item.get("text") or ""
                start = item.get("start")
                end = item.get("end")
                if token and start is not None and end is not None:
                    words.append(WordTimestamp(word=str(token).strip(), start=float(start), end=float(end)))
        return words

    timestep = None
    if isinstance(payload, dict):
        timestep = payload.get("timestep")
    if timestep is None:
        timestep = getattr(output, "timestep", None)

    if isinstance(timestep, dict):
        word_items = timestep.get("word") or []
        time_stride = None
        if model is not None:
            try:
                time_stride = 8 * float(model.cfg.preprocessor.window_stride)
            except Exception:
                time_stride = None

        for item in word_items:
            if not isinstance(item, dict):
                continue
            token = item.get("word") or item.get("char") or ""
            start = item.get("start")
            end = item.get("end")
            if start is None:
                start = item.get("start_time")
            if end is None:
                end = item.get("end_time")
            if start is None:
                start = item.get("start_offset")
            if end is None:
                end = item.get("end_offset")
            if not token or start is None or end is None:
                continue
            start_f = float(start)
            end_f = float(end)
            if time_stride is not None and ("start_offset" in item or "end_offset" in item):
                start_f *= time_stride
                end_f *= time_stride
            words.append(WordTimestamp(word=str(token).strip(), start=start_f, end=end_f))
        return words

    raise RuntimeError("Unable to extract Parakeet word timestamps from transcription output.")


@app.local_entrypoint()
def main(action: str = "gpu-check", audio: str = "") -> None:
    """Local Modal entrypoint for smoke checks and volume-backed uploads.

    Examples:
        modal run modal_app.py --action gpu-check
        modal run modal_app.py --action cache-models
        modal run modal_app.py --action upload-audio --audio data/example.m4a
    """
    if action == "gpu-check":
        print(gpu_check.remote())
        return
    if action == "cache-models":
        for path in download_models.remote():
            print(path)
        return
    if action == "list-cache":
        for path in list_cached_files.remote():
            print(path)
        return
    if action == "upload-audio":
        if not audio:
            raise SystemExit("--audio is required for action=upload-audio")
        uploaded = upload_audio(audio)
        print(uploaded.model_dump_json(indent=2))
        return
    if action == "list-cache":
        for path in list_cached_files.remote():
            print(path)
        return
    raise SystemExit(f"Unknown action: {action}")
