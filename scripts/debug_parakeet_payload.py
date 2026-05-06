from __future__ import annotations

import json
import sys

import modal

from diarizer.engines.modal_remote import audio_id_for_file


def main() -> None:
    audio = sys.argv[1]
    audio_id = audio_id_for_file(audio)
    fn = modal.Function.from_name("diarizer", "debug_parakeet_asr")
    payload = fn.remote(audio_id=audio_id, model_id="nvidia/parakeet-tdt-1.1b", language="en")
    print(json.dumps(payload, indent=2)[:12000])


if __name__ == "__main__":
    main()
