from typing import get_args

import pytest
from pydantic import ValidationError

from vocalbin.openai.realtime import (
    SemanticVadConfig,
    SessionType,
    TranscriptionConfig,
    TranslationConfig,
    TranslationLanguage,
    events,
)


def test_realtime_transcription_config_defaults_and_normalizes_language() -> None:
    config = TranscriptionConfig(language=" de ")

    assert config.model == "gpt-realtime-whisper"
    assert config.language == "de"
    assert config.delay == "medium"
    assert config.noise_reduction == "far_field"
    assert config.turn_detection is None
    assert config.include_logprobs is False
    assert TranscriptionConfig(language=None).language is None
    assert set(get_args(SessionType.__value__)) == {
        "transcription",
        "translation",
    }


def test_realtime_transcription_config_rejects_invalid_fields() -> None:
    with pytest.raises(ValidationError, match="language must not be blank"):
        TranscriptionConfig(language=" ")

    with pytest.raises(ValidationError, match="Extra inputs"):
        TranscriptionConfig.model_validate({"unknown": True})


def test_semantic_vad_config_defaults_and_validates_eagerness() -> None:
    config = SemanticVadConfig()

    assert config.type == "semantic_vad"
    assert config.eagerness == "medium"

    with pytest.raises(ValidationError):
        SemanticVadConfig(eagerness="auto")


def test_realtime_whisper_rejects_turn_detection() -> None:
    with pytest.raises(ValidationError, match="does not support turn detection"):
        TranscriptionConfig(turn_detection=SemanticVadConfig())

    config = TranscriptionConfig(
        model="gpt-4o-transcribe",
        turn_detection=SemanticVadConfig(),
    )

    assert config.model == "gpt-4o-transcribe"


def test_realtime_configs_reject_unknown_model_ids() -> None:
    with pytest.raises(ValidationError):
        TranscriptionConfig(model="transcription-future")

    with pytest.raises(ValidationError):
        TranslationConfig(
            model="translation-future",
            target_language="de",
        )


def test_realtime_translation_config_supports_documented_languages() -> None:
    config = TranslationConfig(
        target_language="de",
        noise_reduction=None,
        include_source_transcript=False,
    )

    assert config.model == "gpt-realtime-translate"
    assert config.target_language == "de"
    assert config.noise_reduction is None
    assert config.include_source_transcript is False
    assert set(get_args(TranslationLanguage.__value__)) == {
        "zh",
        "en",
        "fr",
        "de",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "pt",
        "ru",
        "es",
        "vi",
    }

    with pytest.raises(ValidationError):
        TranslationConfig(target_language="nl")


def test_realtime_error_details_have_readable_string() -> None:
    error = events.ErrorDetails(type="server_error", message="Unavailable")

    assert str(error) == "[server_error] Unavailable"
