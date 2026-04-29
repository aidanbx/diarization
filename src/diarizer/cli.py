"""Diarization CLI.

    diarize run <audio>              # full pipeline
    diarize stage <name> <audio>     # run one stage (requires prior stages cached)
    diarize clean <audio>            # delete cached artifacts for an audio file
    diarize download-models          # download model files
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Speaker diarization pipeline", add_completion=False)
console = Console()

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "default.yaml"

STAGES = ["vad", "segment", "embed", "cluster", "asr", "align"]


def _load_pipeline(config: Path) -> "Pipeline":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from diarizer.pipeline import Pipeline, PipelineConfig
    cfg = PipelineConfig.from_yaml(config)
    return Pipeline(cfg)


@app.command()
def run(
    audio: Path = typer.Argument(..., help="Path to audio file"),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Config YAML"),
    skip_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached stage outputs"),
) -> None:
    """Run the full diarization pipeline on an audio file."""
    if not audio.exists():
        console.print(f"[red]File not found: {audio}[/red]")
        raise typer.Exit(1)
    pipeline = _load_pipeline(config)
    transcript = pipeline.run(audio, skip_cache=skip_cache)
    console.print(f"[green]Done. {len(transcript.turns)} speaker turns.[/green]")


@app.command()
def stage(
    name: str = typer.Argument(..., help=f"Stage name: {', '.join(STAGES)}"),
    audio: Path = typer.Argument(..., help="Path to audio file"),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Run a single pipeline stage (prior stages must be cached)."""
    if name not in STAGES:
        console.print(f"[red]Unknown stage '{name}'. Choose from: {', '.join(STAGES)}[/red]")
        raise typer.Exit(1)
    pipeline = _load_pipeline(config)
    result = pipeline.run_stage(name, audio)
    console.print(f"[green]Stage '{name}' complete.[/green]")


@app.command()
def clean(
    audio: Path = typer.Argument(..., help="Path to audio file"),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Delete cached stage artifacts for an audio file."""
    import shutil, hashlib
    from diarizer.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    # Compute hash of the file
    h = hashlib.sha256()
    with open(audio, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    file_hash = h.hexdigest()[:16]
    cache_dir = Path(cfg.cache_dir) / file_hash
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        console.print(f"[green]Cleared cache for {audio.name} ({file_hash})[/green]")
    else:
        console.print(f"No cache found for {audio.name}")


@app.command(name="download-models")
def download_models() -> None:
    """Download all required model files."""
    import subprocess, sys
    script = Path(__file__).parent.parent.parent / "scripts" / "download_models.py"
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    app()
