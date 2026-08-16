from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

type TranscriptionModel = Literal[
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-transcribe",
    "gpt-4o-transcribe-diarize",
    "gpt-realtime-whisper",
]
type TranslationModel = Literal["gpt-realtime-translate"]

type SessionType = Literal["transcription", "translation"]

type TranscriptionDelay = Literal["minimal", "low", "medium", "high", "xhigh"]

type NoiseReduction = Literal["near_field", "far_field"]

type TranslationLanguage = Literal[
    "zh", "en", "fr", "de", "hi", "id", "it", "ja", "ko", "pt", "ru", "es", "vi"
]

type VadEagerness = Literal["low", "medium", "high"]


class SemanticVadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["semantic_vad"] = "semantic_vad"
    eagerness: VadEagerness = "medium"


class TranscriptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: TranscriptionModel = "gpt-realtime-whisper"
    language: str | None = None
    delay: TranscriptionDelay = "medium"
    noise_reduction: NoiseReduction | None = "far_field"
    turn_detection: SemanticVadConfig | None = None
    include_logprobs: bool = False

    @model_validator(mode="after")
    def validate_turn_detection_support(self) -> Self:
        if self.model == "gpt-realtime-whisper" and self.turn_detection is not None:
            raise ValueError(
                "gpt-realtime-whisper does not support turn detection; "
                "use gpt-4o-transcribe or set turn_detection=None"
            )
        return self

    @field_validator("language")
    @classmethod
    def language_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("language must not be blank")
        return normalized


class TranslationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: TranslationModel = "gpt-realtime-translate"
    target_language: TranslationLanguage
    noise_reduction: NoiseReduction | None = "far_field"
    include_source_transcript: bool = True
