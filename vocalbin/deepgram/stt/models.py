from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpeechToTextModel(StrEnum):
    NOVA_3 = "nova-3"
    NOVA_3_GENERAL = "nova-3-general"
    NOVA_3_MEDICAL = "nova-3-medical"
    NOVA_2 = "nova-2"


class SpeechToTextEncoding(StrEnum):
    LINEAR16 = "linear16"
    FLAC = "flac"
    MULAW = "mulaw"
    ALAW = "alaw"
    OPUS = "opus"
    SPEEX = "speex"
    G729 = "g729"


class FluxModel(StrEnum):
    FLUX_GENERAL_EN = "flux-general-en"
    FLUX_GENERAL_MULTI = "flux-general-multi"


StreamingSpeechToTextModel = FluxModel


class StreamingSpeechToTextEncoding(StrEnum):
    LINEAR16 = "linear16"
    LINEAR32 = "linear32"
    MULAW = "mulaw"
    ALAW = "alaw"
    OPUS = "opus"
    OGG_OPUS = "ogg-opus"


class StreamingRedaction(StrEnum):
    NUMBERS = "numbers"
    AGGRESSIVE_NUMBERS = "aggressive_numbers"


MULTILINGUAL_MODEL = FluxModel.FLUX_GENERAL_MULTI

LANGUAGE_HINTS = frozenset({"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"})


def _keyterms_must_not_be_blank(value: list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if any(not keyterm.strip() for keyterm in value):
        raise ValueError("keyterms must not contain blank values")
    return value


class SpeechToTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: SpeechToTextModel | str = SpeechToTextModel.NOVA_3
    language: str | None = None
    encoding: SpeechToTextEncoding | None = None
    keyterms: list[str] | None = Field(default=None, serialization_alias="keyterm")
    smart_format: bool | None = None
    punctuate: bool | None = None
    paragraphs: bool | None = None
    utterances: bool | None = None
    diarize: bool | None = None
    filler_words: bool | None = None
    numerals: bool | None = None
    profanity_filter: bool | None = None
    detect_language: bool | None = None

    _keyterms_must_not_be_blank = field_validator("keyterms")(
        _keyterms_must_not_be_blank
    )

    @field_validator("language")
    @classmethod
    def language_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("language must not be blank")
        return normalized

    @model_validator(mode="after")
    def keyterms_require_nova_3(self) -> Self:
        if self.keyterms is not None and not str(self.model).startswith("nova-3"):
            raise ValueError("keyterms are supported only by nova-3 models")
        return self


class SpeechToTextResponse(BaseModel):
    text: str
    model: SpeechToTextModel | str
    request_id: str | None = None
    confidence: float | None = None
    detected_language: str | None = None
    raw: dict[str, Any]


class StreamingSpeechToTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: FluxModel | str = FluxModel.FLUX_GENERAL_EN
    encoding: StreamingSpeechToTextEncoding = StreamingSpeechToTextEncoding.LINEAR16
    sample_rate: int = Field(default=16000, gt=0)
    keyterms: list[str] | None = Field(default=None, serialization_alias="keyterm")
    language_hints: list[str] | None = Field(
        default=None, serialization_alias="language_hint"
    )
    eot_threshold: float | None = Field(default=None, ge=0.5, le=0.9)
    eager_eot_threshold: float | None = Field(default=None, ge=0.3, le=0.9)
    eot_timeout_ms: int | None = Field(default=None, ge=500, le=60000)
    numerals: bool | None = None
    profanity_filter: bool | None = None
    redact: StreamingRedaction | None = None

    _keyterms_must_not_be_blank = field_validator("keyterms")(
        _keyterms_must_not_be_blank
    )

    @field_validator("language_hints")
    @classmethod
    def language_hints_must_be_supported(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        # Deepgram accepts locale subtags such as `pt-BR`; the base tag decides.
        unsupported = sorted(
            {hint for hint in value if hint.split("-")[0].lower() not in LANGUAGE_HINTS}
        )
        if unsupported:
            raise ValueError(
                f"unsupported language hints: {', '.join(unsupported)}",
            )
        return value

    @model_validator(mode="after")
    def language_hints_require_the_multilingual_model(self) -> Self:
        if self.language_hints is not None and str(self.model) != MULTILINGUAL_MODEL:
            raise ValueError(
                f"language_hints are supported only by {MULTILINGUAL_MODEL}",
            )
        return self

    @model_validator(mode="after")
    def eager_threshold_must_not_exceed_end_threshold(self) -> Self:
        eot = self.eot_threshold if self.eot_threshold is not None else 0.7
        if self.eager_eot_threshold is not None and self.eager_eot_threshold > eot:
            raise ValueError(
                "eager_eot_threshold must not exceed eot_threshold",
            )
        return self
