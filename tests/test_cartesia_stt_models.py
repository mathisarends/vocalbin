import pytest
from pydantic import TypeAdapter, ValidationError

from vocalbin.cartesia import (
    SpeechToTextConfig,
    SpeechToTextEncoding,
    SpeechToTextModel,
    events,
)


def test_cartesia_stt_config_defaults_and_serialization() -> None:
    config = SpeechToTextConfig(keyterms=None)

    assert config.model == SpeechToTextModel.INK_2
    assert config.encoding == SpeechToTextEncoding.PCM_S16LE
    assert config.to_cartesia_params() == {
        "model": "ink-2",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }
    assert config.keyterms is None


def test_cartesia_stt_config_serializes_every_option() -> None:
    config = SpeechToTextConfig(
        model="ink-future",
        encoding=SpeechToTextEncoding.PCM_F32LE,
        sample_rate=48000,
        keyterms=["vocalbin", "Ink 2"],
        turn_start_threshold=0.9,
        turn_eager_end_threshold=0.5,
        turn_end_threshold=0.1,
        turn_end_timeout_ms=8000,
    )

    assert config.to_cartesia_params() == {
        "model": "ink-future",
        "encoding": "pcm_f32le",
        "sample_rate": 48000,
        "keyterm": ["vocalbin", "Ink 2"],
        "turn_start_threshold": 0.9,
        "turn_eager_end_threshold": 0.5,
        "turn_end_threshold": 0.1,
        "turn_end_timeout_ms": 8000,
    }


@pytest.mark.parametrize(
    "values",
    [
        {"sample_rate": 0},
        {"keyterms": [" "]},
        {"keyterms": ["x"] * 101},
        {"keyterms": ["x" * 1201]},
        {"turn_start_threshold": 0.49},
        {"turn_eager_end_threshold": 0.61},
        {"turn_end_threshold": 0.01},
        {"turn_end_timeout_ms": 639},
        {"unknown": True},
    ],
)
def test_cartesia_stt_config_rejects_invalid_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SpeechToTextConfig(**values)


def test_cartesia_stt_config_rejects_misordered_thresholds() -> None:
    with pytest.raises(ValidationError, match="start > eager_end > end"):
        SpeechToTextConfig(
            turn_start_threshold=0.5,
            turn_eager_end_threshold=0.5,
        )


def test_cartesia_stt_event_union_is_discriminated() -> None:
    event = TypeAdapter(events.Event).validate_python(
        {
            "type": "turn.end",
            "request_id": "request-id",
            "transcript": "Hello",
        }
    )

    assert isinstance(event, events.TurnEnd)
