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


class StreamingSpeechToTextModel(StrEnum):
    FLUX_GENERAL_EN = "flux-general-en"
    FLUX_GENERAL_MULTI = "flux-general-multi"


class StreamingSpeechToTextEncoding(StrEnum):
    LINEAR16 = "linear16"
    MULAW = "mulaw"
    ALAW = "alaw"


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

    model: StreamingSpeechToTextModel | str = StreamingSpeechToTextModel.FLUX_GENERAL_EN
    encoding: StreamingSpeechToTextEncoding = StreamingSpeechToTextEncoding.LINEAR16
    sample_rate: int = Field(default=16000, gt=0)
    keyterms: list[str] | None = Field(default=None, serialization_alias="keyterm")
    eot_threshold: float | None = Field(default=None, ge=0.5, le=0.9)
    eager_eot_threshold: float | None = Field(default=None, ge=0.3, le=0.9)
    eot_timeout_ms: int | None = Field(default=None, ge=500, le=60000)

    _keyterms_must_not_be_blank = field_validator("keyterms")(
        _keyterms_must_not_be_blank
    )

    @model_validator(mode="after")
    def eager_threshold_must_not_exceed_end_threshold(self) -> Self:
        eot = self.eot_threshold if self.eot_threshold is not None else 0.7
        if self.eager_eot_threshold is not None and self.eager_eot_threshold > eot:
            raise ValueError(
                "eager_eot_threshold must not exceed eot_threshold",
            )
        return self
