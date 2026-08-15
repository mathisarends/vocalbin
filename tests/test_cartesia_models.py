import pytest
from pydantic import ValidationError

from vocalbin.cartesia import (
    CartesiaAudioEncoding,
    CartesiaGenerationConfig,
    CartesiaMp3OutputFormat,
    CartesiaRawOutputFormat,
    CartesiaTextToSpeechConfig,
    CartesiaWavOutputFormat,
)


def test_cartesia_config_defaults_and_serialization() -> None:
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    assert config.to_cartesia_params() == {
        "model_id": "sonic-3.5",
        "voice": {"mode": "id", "id": "voice-id"},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    }


def test_cartesia_config_forwards_unknown_model_ids() -> None:
    config = CartesiaTextToSpeechConfig(
        voice_id="voice-id",
        model="sonic-future",
    )

    assert config.model == "sonic-future"
    assert config.to_cartesia_params()["model_id"] == "sonic-future"


def test_cartesia_config_serializes_optional_parameters() -> None:
    config = CartesiaTextToSpeechConfig(
        voice_id="voice-id",
        output_format=CartesiaWavOutputFormat(
            encoding=CartesiaAudioEncoding.PCM_F32LE,
            sample_rate=48000,
        ),
        language=" DE ",
        generation_config=CartesiaGenerationConfig(
            emotion="happy", speed=1.2, volume=0.8
        ),
        pronunciation_dict_id="dictionary-id",
    )

    assert config.language == "de"
    assert config.to_cartesia_params() == {
        "model_id": "sonic-3.5",
        "voice": {"mode": "id", "id": "voice-id"},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 48000,
        },
        "language": "de",
        "generation_config": {
            "emotion": "happy",
            "speed": 1.2,
            "volume": 0.8,
        },
        "pronunciation_dict_id": "dictionary-id",
    }


def test_cartesia_output_formats_have_typed_defaults() -> None:
    assert CartesiaRawOutputFormat().container == "raw"
    assert CartesiaWavOutputFormat().sample_rate == 44100
    assert CartesiaMp3OutputFormat().bit_rate == 128000


def test_cartesia_generation_config_rejects_blank_emotion() -> None:
    with pytest.raises(ValidationError, match="emotion must not be blank"):
        CartesiaGenerationConfig(emotion="  ")


@pytest.mark.parametrize("field", ["voice_id", "pronunciation_dict_id"])
def test_cartesia_identifiers_reject_blank_values(field: str) -> None:
    values = {"voice_id": "voice-id", field: " "}

    with pytest.raises(ValidationError, match="identifier must not be blank"):
        CartesiaTextToSpeechConfig(**values)


@pytest.mark.parametrize("language", ["", "eng", "d1", "dÃ©"])
def test_cartesia_language_rejects_non_iso_639_1_values(language: str) -> None:
    with pytest.raises(ValidationError, match="language must be an ISO-639-1 code"):
        CartesiaTextToSpeechConfig(voice_id="voice-id", language=language)


def test_cartesia_optional_values_accept_none() -> None:
    config = CartesiaTextToSpeechConfig(
        voice_id="voice-id",
        language=None,
        pronunciation_dict_id=None,
    )
    generation = CartesiaGenerationConfig(emotion=None)

    assert config.language is None
    assert generation.to_cartesia_params() == {}


@pytest.mark.parametrize(
    "values",
    [
        {"generation_config": {"speed": 0.59}},
        {"generation_config": {"volume": 2.01}},
        {"max_buffer_delay_ms": -1},
        {"timeout": 0},
        {"unknown": True},
    ],
)
def test_cartesia_config_rejects_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CartesiaTextToSpeechConfig(voice_id="voice-id", **values)


def test_cartesia_output_format_is_discriminated_and_validated() -> None:
    config = CartesiaTextToSpeechConfig.model_validate(
        {
            "voice_id": "voice-id",
            "output_format": {
                "container": "mp3",
                "sample_rate": 24000,
                "bit_rate": 64000,
            },
        }
    )

    assert isinstance(config.output_format, CartesiaMp3OutputFormat)
    with pytest.raises(ValidationError):
        CartesiaRawOutputFormat(sample_rate=12345)
