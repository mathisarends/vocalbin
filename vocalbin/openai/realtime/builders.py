from __future__ import annotations

from typing import TYPE_CHECKING, Self

from vocalbin.openai.realtime.models import (
    RealtimeNoiseReduction,
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionDelay,
    RealtimeTranscriptionModel,
    RealtimeTranslationConfig,
    RealtimeTranslationLanguage,
    RealtimeTranslationModel,
    RealtimeVadEagerness,
    SemanticVadConfig,
)
from vocalbin.openai.realtime.ports import AudioInput, RealtimeProvider

if TYPE_CHECKING:
    from vocalbin.openai.realtime.clients import (
        Transcriber,
        Translator,
    )


class RealtimeTranscriberBuilder:
    def __init__(self) -> None:
        defaults = RealtimeTranscriptionConfig()
        self._model = defaults.model
        self._language = defaults.language
        self._delay = defaults.delay
        self._noise_reduction = defaults.noise_reduction
        self._turn_detection = defaults.turn_detection
        self._include_logprobs = defaults.include_logprobs
        self._audio_input: AudioInput | None = None
        self._provider: RealtimeProvider | None = None
        self._api_key: str | None = None
        self._safety_identifier: str | None = None

    def model(self, model: RealtimeTranscriptionModel) -> Self:
        self._model = model
        return self

    def language(self, language: str | None) -> Self:
        self._language = language
        return self

    def delay(self, delay: RealtimeTranscriptionDelay) -> Self:
        self._delay = delay
        return self

    def noise_reduction(self, noise_reduction: RealtimeNoiseReduction | None) -> Self:
        self._noise_reduction = noise_reduction
        return self

    def semantic_vad(self, eagerness: RealtimeVadEagerness = "medium") -> Self:
        self._turn_detection = SemanticVadConfig(eagerness=eagerness)
        return self

    def turn_detection(self, turn_detection: SemanticVadConfig | None) -> Self:
        self._turn_detection = turn_detection
        return self

    def include_logprobs(self, include: bool = True) -> Self:
        self._include_logprobs = include
        return self

    def audio_input(self, audio_input: AudioInput) -> Self:
        self._audio_input = audio_input
        return self

    def provider(self, provider: RealtimeProvider) -> Self:
        self._provider = provider
        return self

    def api_key(self, api_key: str) -> Self:
        self._api_key = api_key
        return self

    def safety_identifier(self, safety_identifier: str) -> Self:
        self._safety_identifier = safety_identifier
        return self

    def build(self) -> Transcriber:
        from vocalbin.openai.realtime.clients import Transcriber

        config = RealtimeTranscriptionConfig(
            model=self._model,
            language=self._language,
            delay=self._delay,
            noise_reduction=self._noise_reduction,
            turn_detection=self._turn_detection,
            include_logprobs=self._include_logprobs,
        )
        return Transcriber(
            config,
            audio_input=self._audio_input,
            provider=self._provider,
            api_key=self._api_key,
            safety_identifier=self._safety_identifier,
        )


class RealtimeTranslatorBuilder:
    def __init__(self) -> None:
        self._model: RealtimeTranslationModel = "gpt-realtime-translate"
        self._target_language: RealtimeTranslationLanguage | None = None
        self._noise_reduction: RealtimeNoiseReduction | None = "far_field"
        self._include_source_transcript = True
        self._audio_input: AudioInput | None = None
        self._provider: RealtimeProvider | None = None
        self._api_key: str | None = None
        self._safety_identifier: str | None = None

    def model(self, model: RealtimeTranslationModel) -> Self:
        self._model = model
        return self

    def target_language(self, language: RealtimeTranslationLanguage) -> Self:
        self._target_language = language
        return self

    def noise_reduction(self, noise_reduction: RealtimeNoiseReduction | None) -> Self:
        self._noise_reduction = noise_reduction
        return self

    def include_source_transcript(self, include: bool = True) -> Self:
        self._include_source_transcript = include
        return self

    def audio_input(self, audio_input: AudioInput) -> Self:
        self._audio_input = audio_input
        return self

    def provider(self, provider: RealtimeProvider) -> Self:
        self._provider = provider
        return self

    def api_key(self, api_key: str) -> Self:
        self._api_key = api_key
        return self

    def safety_identifier(self, safety_identifier: str) -> Self:
        self._safety_identifier = safety_identifier
        return self

    def build(self) -> Translator:
        from vocalbin.openai.realtime.clients import Translator

        if self._target_language is None:
            raise ValueError("target_language must be configured before build()")
        config = RealtimeTranslationConfig(
            model=self._model,
            target_language=self._target_language,
            noise_reduction=self._noise_reduction,
            include_source_transcript=self._include_source_transcript,
        )
        return Translator(
            config,
            audio_input=self._audio_input,
            provider=self._provider,
            api_key=self._api_key,
            safety_identifier=self._safety_identifier,
        )
