"""Download all required model files to models/ directory.

Run once before using the pipeline:
    conda activate diarization
    python scripts/download_models.py

Idempotent — skips already-downloaded files.
"""
from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# sherpa-onnx GitHub releases base
SHERPA_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

MODELS = [
    {
        "name": "pyannote-segmentation-3.0",
        "url": f"{SHERPA_BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "archive": "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "check_file": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    },
    {
        # English CAM++ model (voxceleb-trained — good for English sessions)
        # tag has a typo: "recongition" not "recognition" — that's the real tag
        "name": "CAM++ speaker embeddings (English)",
        "url": f"{SHERPA_BASE}/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "archive": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "check_file": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    },
]


def download(url: str, dest: Path) -> None:
    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def extract(archive: Path) -> None:
    name = archive.name
    if name.endswith(".tar.bz2") or name.endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            tf.extractall(MODELS_DIR)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(MODELS_DIR)
    # plain .onnx files: nothing to extract


def main() -> None:
    for model in MODELS:
        check = MODELS_DIR / model["check_file"]
        if check.exists():
            print(f"[ok] {model['name']} already present")
            continue
        print(f"[dl] {model['name']}")
        archive = MODELS_DIR / model["archive"]
        download(model["url"], archive)
        if archive.suffix not in (".onnx",):
            extract(archive)
            archive.unlink()
        print(f"[ok] {model['name']}")

    print("\nAll models ready in", MODELS_DIR)


if __name__ == "__main__":
    main()
