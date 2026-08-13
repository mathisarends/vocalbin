import pytest
from pydantic import ValidationError

from vocalbin.openai.realtime import (
    RealtimeErrorDetails,
    RealtimeNoiseReduction,
    RealtimeSessionType,
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionDelay,
    RealtimeTranscriptionModel,
    RealtimeTranslationConfig,
    RealtimeTranslationLanguage,
    RealtimeTranslationModel,
    SemanticVadConfig,
)


def test_realtime_transcription_config_defaults_and_normalizes_language() -> None:
    config = RealtimeTranscriptionConfig(language=" de ")

    assert config.model == RealtimeTranscriptionModel.GPT_REALTIME_WHISPER
    assert config.language == "de"
    assert config.delay == RealtimeTranscriptionDelay.MEDIUM
    assert config.noise_reduction == RealtimeNoiseReduction.FAR_FIELD
    assert config.turn_detection is None
    assert config.include_logprobs is False
    assert RealtimeTranscriptionConfig(language=None).language is None
    assert {session_type.value for session_type in RealtimeSessionType} == {
        "transcription",
        "translation",
    }


def test_realtime_transcription_config_rejects_invalid_fields() -> None:
    with pytest.raises(ValidationError, match="language must not be blank"):
        RealtimeTranscriptionConfig(language=" ")

    with pytest.raises(ValidationError, match="Extra inputs"):
        RealtimeTranscriptionConfig.model_validate({"unknown": True})


def test_semantic_vad_config_defaults_and_validates_eagerness() -> None:
    config = SemanticVadConfig()

    assert config.type == "semantic_vad"
    assert config.eagerness == "medium"

    with pytest.raises(ValidationError):
        SemanticVadConfig(eagerness="auto")


def test_realtime_whisper_rejects_turn_detection() -> None:
    with pytest.raises(ValidationError, match="does not support turn detection"):
        RealtimeTranscriptionConfig(turn_detection=SemanticVadConfig())

    config = RealtimeTranscriptionConfig(
        model=RealtimeTranscriptionModel.GPT_4O_TRANSCRIBE,
        turn_detection=SemanticVadConfig(),
    )

    assert config.model == RealtimeTranscriptionModel.GPT_4O_TRANSCRIBE


def test_realtime_configs_forward_unknown_model_ids() -> None:
    transcription = RealtimeTranscriptionConfig(model="transcription-future")
    translation = RealtimeTranslationConfig(
        model="translation-future",
        target_language=RealtimeTranslationLanguage.GERMAN,
    )

    assert transcription.model == "transcription-future"
    assert translation.model == "translation-future"


def test_realtime_translation_config_supports_documented_languages() -> None:
    config = RealtimeTranslationConfig(
        target_language=RealtimeTranslationLanguage.GERMAN,
        noise_reduction=None,
        include_source_transcript=False,
    )

    assert config.model == RealtimeTranslationModel.GPT_REALTIME_TRANSLATE
    assert config.target_language == "de"
    assert isinstance(config.target_language, RealtimeTranslationLanguage)
    assert config.noise_reduction is None
    assert config.include_source_transcript is False
    assert {language.value for language in RealtimeTranslationLanguage} == {
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
        RealtimeTranslationConfig(target_language="nl")


def test_realtime_error_details_have_readable_string() -> None:
    error = RealtimeErrorDetails(type="server_error", message="Unavailable")

    assert str(error) == "[server_error] Unavailable"
