from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CartesiaSpeechToTextModel(StrEnum):
    INK_2 = "ink-2"


class CartesiaSpeechToTextEncoding(StrEnum):
    PCM_S16LE = "pcm_s16le"
    PCM_S32LE = "pcm_s32le"
    PCM_F16LE = "pcm_f16le"
    PCM_F32LE = "pcm_f32le"
    PCM_MULAW = "pcm_mulaw"
    PCM_ALAW = "pcm_alaw"


class CartesiaSpeechToTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: CartesiaSpeechToTextModel | str = CartesiaSpeechToTextModel.INK_2
    encoding: CartesiaSpeechToTextEncoding = CartesiaSpeechToTextEncoding.PCM_S16LE
    sample_rate: int = Field(default=16000, gt=0)
    keyterms: list[str] | None = Field(default=None, max_length=100)
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

    def to_cartesia_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": (
                self.model.value
                if isinstance(self.model, CartesiaSpeechToTextModel)
                else self.model
            ),
            "encoding": self.encoding.value,
            "sample_rate": self.sample_rate,
        }
        if self.keyterms is not None:
            params["keyterm"] = self.keyterms
        for name in (
            "turn_start_threshold",
            "turn_eager_end_threshold",
            "turn_end_threshold",
            "turn_end_timeout_ms",
        ):
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        return params


class CartesiaSpeechToTextConnected(BaseModel):
    type: Literal["connected"] = "connected"
    request_id: str


class CartesiaSpeechToTextTurnStart(BaseModel):
    type: Literal["turn.start"] = "turn.start"
    request_id: str


class CartesiaSpeechToTextTurnUpdate(BaseModel):
    type: Literal["turn.update"] = "turn.update"
    request_id: str
    transcript: str


class CartesiaSpeechToTextTurnEagerEnd(BaseModel):
    type: Literal["turn.eager_end"] = "turn.eager_end"
    request_id: str
    transcript: str


class CartesiaSpeechToTextTurnResume(BaseModel):
    type: Literal["turn.resume"] = "turn.resume"
    request_id: str


class CartesiaSpeechToTextTurnEnd(BaseModel):
    type: Literal["turn.end"] = "turn.end"
    request_id: str
    transcript: str


type CartesiaSpeechToTextEvent = Annotated[
    CartesiaSpeechToTextConnected
    | CartesiaSpeechToTextTurnStart
    | CartesiaSpeechToTextTurnUpdate
    | CartesiaSpeechToTextTurnEagerEnd
    | CartesiaSpeechToTextTurnResume
    | CartesiaSpeechToTextTurnEnd,
    Field(discriminator="type"),
]
