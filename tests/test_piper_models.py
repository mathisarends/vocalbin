import pytest
from pydantic import ValidationError

from vocalbin.piper import TextToSpeechConfig


def test_model_dump_omits_unset_fields() -> None:
    config = TextToSpeechConfig(speaker_id=2)

    assert config.model_dump(exclude_none=True) == {"speaker_id": 2}


def test_model_dump_includes_all_set_fields() -> None:
    config = TextToSpeechConfig(
        speaker_id=1, length_scale=1.2, noise_scale=0.5, noise_w_scale=0.7
    )

    assert config.model_dump(exclude_none=True) == {
        "speaker_id": 1,
        "length_scale": 1.2,
        "noise_scale": 0.5,
        "noise_w_scale": 0.7,
    }


def test_config_rejects_negative_speaker_id() -> None:
    with pytest.raises(ValidationError):
        TextToSpeechConfig(speaker_id=-1)
