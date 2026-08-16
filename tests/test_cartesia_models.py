import pytest
from pydantic import ValidationError

from vocalbin.cartesia import (
    AudioEncoding,
    GenerationConfig,
    Mp3OutputFormat,
    RawOutputFormat,
    TextToSpeechConfig,
    Voice,
    WavOutputFormat,
)


def test_cartesia_config_defaults_and_serialization() -> None:
    config = TextToSpeechConfig(voice_id="voice-id")

    assert config.model_dump(exclude_none=True, mode="json", by_alias=True) == {
        "model_id": "sonic-3.5",
        "voice_id": "voice-id",
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    }


def test_cartesia_config_accepts_generated_voice() -> None:
    voice = next(iter(Voice))
    config = TextToSpeechConfig(voice_id=voice)

    assert config.voice_id == voice
    assert config.model_dump(mode="json")["voice_id"] == voice.value


def test_cartesia_config_forwards_unknown_model_ids() -> None:
    config = TextToSpeechConfig(
        voice_id="voice-id",
        model="sonic-future",
    )

    assert config.model == "sonic-future"
    assert config.model_dump(mode="json", by_alias=True)["model_id"] == "sonic-future"


def test_cartesia_config_serializes_optional_parameters() -> None:
    config = TextToSpeechConfig(
        voice_id="voice-id",
        output_format=WavOutputFormat(
            encoding=AudioEncoding.PCM_F32LE,
            sample_rate=48000,
        ),
        language=" DE ",
        generation_config=GenerationConfig(emotion="happy", speed=1.2, volume=0.8),
        pronunciation_dict_id="dictionary-id",
    )

    assert config.language == "de"
    assert config.model_dump(exclude_none=True, mode="json", by_alias=True) == {
        "voice_id": "voice-id",
        "model_id": "sonic-3.5",
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
    assert RawOutputFormat().container == "raw"
    assert WavOutputFormat().sample_rate == 44100
    assert Mp3OutputFormat().bit_rate == 128000


def test_cartesia_generation_config_rejects_blank_emotion() -> None:
    with pytest.raises(ValidationError, match="emotion must not be blank"):
        GenerationConfig(emotion="  ")


@pytest.mark.parametrize("field", ["voice_id", "pronunciation_dict_id"])
def test_cartesia_identifiers_reject_blank_values(field: str) -> None:
    values = {"voice_id": "voice-id", field: " "}

    with pytest.raises(ValidationError, match="identifier must not be blank"):
        TextToSpeechConfig(**values)


@pytest.mark.parametrize("language", ["", "eng", "d1", "dÃ©"])
def test_cartesia_language_rejects_non_iso_639_1_values(language: str) -> None:
    with pytest.raises(ValidationError, match="language must be an ISO-639-1 code"):
        TextToSpeechConfig(voice_id="voice-id", language=language)


def test_cartesia_optional_values_accept_none() -> None:
    config = TextToSpeechConfig(
        voice_id="voice-id",
        language=None,
        pronunciation_dict_id=None,
    )
    generation = GenerationConfig(emotion=None)

    assert config.language is None
    assert generation.model_dump(exclude_none=True, mode="json") == {}


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
        TextToSpeechConfig(voice_id="voice-id", **values)


def test_cartesia_output_format_is_discriminated_and_validated() -> None:
    config = TextToSpeechConfig.model_validate(
        {
            "voice_id": "voice-id",
            "output_format": {
                "container": "mp3",
                "sample_rate": 24000,
                "bit_rate": 64000,
            },
        }
    )

    assert isinstance(config.output_format, Mp3OutputFormat)
    with pytest.raises(ValidationError):
        RawOutputFormat(sample_rate=12345)
