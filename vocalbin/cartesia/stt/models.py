from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpeechToTextModel(StrEnum):
    INK_2 = "ink-2"


class SpeechToTextEncoding(StrEnum):
    PCM_S16LE = "pcm_s16le"
    PCM_S32LE = "pcm_s32le"
    PCM_F16LE = "pcm_f16le"
    PCM_F32LE = "pcm_f32le"
    PCM_MULAW = "pcm_mulaw"
    PCM_ALAW = "pcm_alaw"


class SpeechToTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: SpeechToTextModel | str = SpeechToTextModel.INK_2
    encoding: SpeechToTextEncoding = SpeechToTextEncoding.PCM_S16LE
    sample_rate: int = Field(default=16000, gt=0)
    keyterms: list[str] | None = Field(
        default=None,
        max_length=100,
        serialization_alias="keyterm",
    )
    turn_start_threshold: float | None = Field(default=None, ge=0.5, le=0.9)
    turn_eager_end_threshold: float | None = Field(default=None, ge=0.3, le=0.6)
    turn_end_threshold: float | None = Field(default=None, ge=0.05, le=0.5)
    turn_end_timeout_ms: float | None = Field(default=None, ge=640, le=11200)

    @field_validator("keyterms")
    @classmethod
    def keyterms_must_be_valid(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not keyterm.strip() for keyterm in value):
            raise ValueError("keyterms must not contain blank values")
        if sum(len(keyterm) for keyterm in value) > 1200:
            raise ValueError("keyterms must not exceed 1200 characters in total")
        return value

    @model_validator(mode="after")
    def thresholds_must_be_ordered(self) -> Self:
        start = self.turn_start_threshold or 0.8
        eager_end = self.turn_eager_end_threshold or 0.4
        end = self.turn_end_threshold or 0.2
        if not start > eager_end > end:
            raise ValueError("turn thresholds must satisfy start > eager_end > end")
        return self
