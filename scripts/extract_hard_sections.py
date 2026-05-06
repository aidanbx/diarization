from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


WINDOWS = [
    ("hard_refs_00h20m00s_to_00h28m30s", "00:20:00", "00:28:30"),
    ("hard_refs_00h58m00s_to_01h03m00s", "00:58:00", "01:03:00"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract known hard transcription windows.")
    parser.add_argument("audio", type=Path, help="Source audio file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hard_sections"),
        help="Directory to write extracted windows",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.audio.stem.replace(" ", "_").replace("#", "")
    for name, start, end in WINDOWS:
        out_path = args.output_dir / f"{stem}_{name}.m4a"
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(args.audio),
            "-ss",
            start,
            "-to",
            end,
            "-c",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        print(out_path)


if __name__ == "__main__":
    main()
