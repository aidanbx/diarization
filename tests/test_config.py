from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_modal_parakeet_config_parses():
    from diarizer.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(Path(__file__).parent.parent / "config" / "modal-diarizen-parakeet.yaml")

    assert cfg.segmenter_engine == "modal_diarizen"
    assert cfg.asr_engine == "modal_parakeet"
    assert cfg.diarization_model == "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    assert cfg.asr_model == "nvidia/parakeet-tdt-0.6b-v3"


def test_elevenlabs_config_parses():
    from diarizer.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(Path(__file__).parent.parent / "config" / "modal-diarizen-elevenlabs.yaml")

    assert cfg.segmenter_engine == "modal_diarizen"
    assert cfg.asr_engine == "elevenlabs_scribe"
    assert cfg.elevenlabs_api_key_env == "ELEVENLABS_API_KEY"
