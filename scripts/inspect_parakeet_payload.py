from __future__ import annotations

import json
import sys

import modal

from diarizer.engines.modal_remote import audio_id_for_file


def main() -> None:
    audio = sys.argv[1]
    audio_id = audio_id_for_file(audio)
    print(f"audio_id={audio_id}")
    fn = modal.Function.from_name("diarizer", "run_parakeet_asr")
    payload = fn.remote(
        audio_id=audio_id,
        model_id="nvidia/parakeet-tdt-1.1b",
        language="en",
        initial_prompt=None,
        hotwords=[],
        skip_cache=True,
    )
    print(f"payload_type={type(payload)}")
    if isinstance(payload, dict):
        print(f"keys={sorted(payload.keys())}")
        words = payload.get("words")
        print(f"words_type={type(words)}")
        if isinstance(words, list):
            print(f"words_len={len(words)}")
            if words:
                print(json.dumps(words[:5], indent=2)[:4000])
        else:
            print(repr(words))
    else:
        print(repr(payload))


if __name__ == "__main__":
    main()
