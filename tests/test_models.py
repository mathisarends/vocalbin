from pathlib import Path

import pytest
from pydantic import ValidationError

from vocalbin.openai import (
    SpeechToTextFormat,
    SpeechToTextModel,
    SpeechToTextRequest,
    TextToSpeechConfig,
    TextToSpeechModel,
    TextToSpeechVoice,
    TimestampGranularity,
)


def test_speech_to_text_requires_exactly_one_audio_source(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    with pytest.raises(ValidationError, match="exactly one"):
        SpeechToTextRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        SpeechToTextRequest(audio=b"audio", audio_path=audio_path)


def test_audio_path_must_exist_and_be_a_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        SpeechToTextRequest(audio_path=tmp_path / "missing.wav")

    with pytest.raises(ValidationError, match="not a file"):
        SpeechToTextRequest(audio_path=tmp_path)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    request = SpeechToTextRequest(audio_path=audio_path)

    assert request.audio_path == audio_path


def test_explicit_none_values_are_accepted_for_optional_fields() -> None:
    request = SpeechToTextRequest(audio=b"audio", audio_path=None, language=None)

    assert request.audio_path is None
    assert request.language is None


def test_filename_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="filename must not be blank"):
        SpeechToTextRequest(audio=b"audio", filename="  ")


def test_language_is_normalized() -> None:
    request = SpeechToTextRequest(audio=b"audio", language=" DE ")

    assert request.language == "de"


@pytest.mark.parametrize("language", ["", "eng", "d3", "äß"])
def test_language_must_be_an_ascii_iso_639_1_code(language: str) -> None:
    with pytest.raises(ValidationError, match="ISO-639-1"):
        SpeechToTextRequest(audio=b"audio", language=language)


def test_timestamp_granularities_require_whisper_verbose_json() -> None:
    with pytest.raises(ValidationError, match="only by whisper-1"):
        SpeechToTextRequest(
            audio=b"audio",
            timestamp_granularities=[TimestampGranularity.WORD],
        )

    request = SpeechToTextRequest(
        audio=b"audio",
        model=SpeechToTextModel.WHISPER_1,
        response_format=SpeechToTextFormat.VERBOSE_JSON,
        timestamp_granularities=[TimestampGranularity.WORD],
    )
    assert request.timestamp_granularities == [TimestampGranularity.WORD]

    with pytest.raises(ValidationError, match="response_format='verbose_json'"):
        SpeechToTextRequest(
            audio=b"audio",
            model=SpeechToTextModel.WHISPER_1,
            timestamp_granularities=[TimestampGranularity.SEGMENT],
        )


def test_logprobs_require_gpt_json_response() -> None:
    with pytest.raises(ValidationError, match="response_format='json'"):
        SpeechToTextRequest(
            audio=b"audio",
            response_format=SpeechToTextFormat.TEXT,
            include=["logprobs"],
        )

    with pytest.raises(ValidationError, match="only by GPT transcription models"):
        SpeechToTextRequest(
            audio=b"audio",
            model=SpeechToTextModel.WHISPER_1,
            include=["logprobs"],
        )

    request = SpeechToTextRequest(audio=b"audio", include=["logprobs"])
    assert request.include == ["logprobs"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt", "names", "does not support prompt"),
        ("include", ["logprobs"], "does not support include/logprobs"),
        (
            "timestamp_granularities",
            [TimestampGranularity.WORD],
            "does not support timestamp_granularities",
        ),
    ],
)
def test_diarization_rejects_unsupported_options(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SpeechToTextRequest.model_validate(
            {
                "audio": b"audio",
                "model": SpeechToTextModel.GPT_4O_TRANSCRIBE_DIARIZE,
                field: value,
            }
        )


@pytest.mark.parametrize(
    ("model", "response_format"),
    [
        (SpeechToTextModel.GPT_4O_TRANSCRIBE, SpeechToTextFormat.SRT),
        (
            SpeechToTextModel.GPT_4O_TRANSCRIBE_DIARIZE,
            SpeechToTextFormat.SRT,
        ),
        (SpeechToTextModel.WHISPER_1, SpeechToTextFormat.DIARIZED_JSON),
    ],
)
def test_speech_models_reject_unsupported_response_formats(
    model: SpeechToTextModel,
    response_format: SpeechToTextFormat,
) -> None:
    with pytest.raises(ValidationError, match="supports only response_format"):
        SpeechToTextRequest(
            audio=b"audio",
            model=model,
            response_format=response_format,
        )


def test_diarization_accepts_supported_defaults() -> None:
    request = SpeechToTextRequest(
        audio=b"audio",
        model=SpeechToTextModel.GPT_4O_TRANSCRIBE_DIARIZE,
    )

    assert request.response_format == SpeechToTextFormat.JSON


def test_openai_params_exclude_local_audio_fields() -> None:
    request = SpeechToTextRequest(audio=b"audio", filename="sample.wav", language="de")

    assert request.model_dump(
        exclude={"audio_path", "audio", "filename"},
        exclude_none=True,
        mode="json",
    ) == {
        "model": "gpt-4o-transcribe",
        "response_format": "json",
        "language": "de",
    }


def test_request_field_constraints_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SpeechToTextRequest(audio=b"")
    with pytest.raises(ValidationError):
        SpeechToTextRequest(audio=b"audio", temperature=1.1)
    with pytest.raises(ValidationError):
        SpeechToTextRequest.model_validate({"audio": b"audio", "unknown": True})

    with pytest.raises(ValidationError):
        TextToSpeechConfig(speed=0.24)
    with pytest.raises(ValidationError):
        TextToSpeechConfig.model_validate({"unknown": True})


def test_requests_forward_unknown_model_and_voice_ids() -> None:
    speech_to_text = SpeechToTextRequest(audio=b"audio", model="stt-future")
    text_to_speech = TextToSpeechConfig(
        model="tts-future",
        voice="voice-future",
    )

    assert speech_to_text.model == "stt-future"
    assert speech_to_text.model_dump(mode="json")["model"] == "stt-future"
    assert text_to_speech.model == "tts-future"
    assert text_to_speech.voice == "voice-future"


def test_legacy_tts_rejects_new_voice_and_instructions() -> None:
    with pytest.raises(ValidationError, match="does not support voice"):
        TextToSpeechConfig(
            model=TextToSpeechModel.TTS_1,
            voice=TextToSpeechVoice.MARIN,
        )

    with pytest.raises(ValidationError, match="does not support instructions"):
        TextToSpeechConfig(
            model=TextToSpeechModel.TTS_1_HD,
            voice=TextToSpeechVoice.ALLOY,
            instructions="Whisper",
        )

    config = TextToSpeechConfig(
        model=TextToSpeechModel.TTS_1,
        voice=TextToSpeechVoice.ALLOY,
    )
    assert config.model == TextToSpeechModel.TTS_1
