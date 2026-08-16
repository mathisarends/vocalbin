from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpeechToTextModel(StrEnum):
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"
    GPT_4O_TRANSCRIBE_DIARIZE = "gpt-4o-transcribe-diarize"
    WHISPER_1 = "whisper-1"


class SpeechToTextFormat(StrEnum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"
    DIARIZED_JSON = "diarized_json"


class TimestampGranularity(StrEnum):
    WORD = "word"
    SEGMENT = "segment"


class SpeechToTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: SpeechToTextModel | str = SpeechToTextModel.GPT_4O_TRANSCRIBE
    response_format: SpeechToTextFormat = SpeechToTextFormat.JSON
    language: str | None = Field(
        default=None,
        description="Optional ISO-639-1 language code such as 'de' or 'en'.",
    )
    prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=1)
    timestamp_granularities: list[TimestampGranularity] | None = None
    include: list[Literal["logprobs"]] | None = None
    chunking_strategy: Literal["auto"] | dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None

    @field_validator("language")
    @classmethod
    def language_must_be_iso_639_1(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 2 or not normalized.isalpha() or not normalized.isascii():
            raise ValueError("language must be an ISO-639-1 code like 'de' or 'en'")
        return normalized

    @model_validator(mode="after")
    def validate_model_capabilities(self) -> Self:
        gpt_transcribe_models = {
            SpeechToTextModel.GPT_4O_TRANSCRIBE,
            SpeechToTextModel.GPT_4O_MINI_TRANSCRIBE,
        }

        if self.model in gpt_transcribe_models:
            supported = {SpeechToTextFormat.JSON, SpeechToTextFormat.TEXT}
            if self.response_format not in supported:
                raise ValueError(
                    f"{self.model} supports only response_format json or text"
                )

        if self.model == SpeechToTextModel.GPT_4O_TRANSCRIBE_DIARIZE:
            supported = {
                SpeechToTextFormat.JSON,
                SpeechToTextFormat.TEXT,
                SpeechToTextFormat.DIARIZED_JSON,
            }
            if self.response_format not in supported:
                raise ValueError(
                    f"{self.model} supports only response_format json, text, "
                    "or diarized_json"
                )
            if self.prompt is not None:
                raise ValueError(f"{self.model} does not support prompt")
            if self.include is not None:
                raise ValueError(f"{self.model} does not support include/logprobs")
            if self.timestamp_granularities is not None:
                raise ValueError(
                    f"{self.model} does not support timestamp_granularities"
                )

        if self.model == SpeechToTextModel.WHISPER_1:
            supported = {
                SpeechToTextFormat.JSON,
                SpeechToTextFormat.TEXT,
                SpeechToTextFormat.SRT,
                SpeechToTextFormat.VERBOSE_JSON,
                SpeechToTextFormat.VTT,
            }
            if self.response_format not in supported:
                raise ValueError(
                    f"{self.model} supports only response_format json, text, srt, "
                    "verbose_json, or vtt"
                )

        if self.timestamp_granularities is not None:
            if self.model != SpeechToTextModel.WHISPER_1:
                raise ValueError(
                    "timestamp_granularities is supported only by whisper-1"
                )
            if self.response_format != SpeechToTextFormat.VERBOSE_JSON:
                raise ValueError(
                    "timestamp_granularities requires response_format='verbose_json'"
                )

        if self.include is not None:
            if self.model not in gpt_transcribe_models:
                raise ValueError(
                    "include/logprobs is supported only by GPT transcription models"
                )
            if self.response_format != SpeechToTextFormat.JSON:
                raise ValueError("include/logprobs requires response_format='json'")

        return self


class SpeechToTextResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    model: SpeechToTextModel | str
    response_format: SpeechToTextFormat
    raw: dict[str, Any] | str
