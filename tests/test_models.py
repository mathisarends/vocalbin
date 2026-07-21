from pathlib import Path

import pytest
from pydantic import ValidationError

from vocalbin import (
    SpeechToTextFormat,
    SpeechToTextModel,
    SpeechToTextRequest,
    TextToSpeechModel,
    TextToSpeechRequest,
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


def test_language_is_normalized() -> None:
    request = SpeechToTextRequest(audio=b"audio", language=" DE ")

    assert request.language == "de"


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


def test_logprobs_require_gpt_json_response() -> None:
    with pytest.raises(ValidationError, match="response_format='json'"):
        SpeechToTextRequest(
            audio=b"audio",
            response_format=SpeechToTextFormat.TEXT,
            include=["logprobs"],
        )


def test_diarization_rejects_unsupported_options() -> None:
    with pytest.raises(ValidationError, match="does not support prompt"):
        SpeechToTextRequest(
            audio=b"audio",
            model=SpeechToTextModel.GPT_4O_TRANSCRIBE_DIARIZE,
            prompt="names",
        )


def test_openai_params_exclude_local_audio_fields() -> None:
    request = SpeechToTextRequest(audio=b"audio", filename="sample.wav", language="de")

    assert request.to_openai_params() == {
        "model": "gpt-4o-transcribe",
        "response_format": "json",
        "language": "de",
    }


def test_legacy_tts_rejects_new_voice_and_instructions() -> None:
    with pytest.raises(ValidationError, match="does not support voice"):
        TextToSpeechRequest(
            text="Hello",
            model=TextToSpeechModel.TTS_1,
            voice=TextToSpeechVoice.MARIN,
        )

    with pytest.raises(ValidationError, match="does not support instructions"):
        TextToSpeechRequest(
            text="Hello",
            model=TextToSpeechModel.TTS_1_HD,
            voice=TextToSpeechVoice.ALLOY,
            instructions="Whisper",
        )
