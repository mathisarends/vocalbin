from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PiperTextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: int | None = Field(default=None, ge=0)
    length_scale: float | None = Field(default=None, gt=0)
    noise_scale: float | None = Field(default=None, ge=0)
    noise_w_scale: float | None = Field(default=None, ge=0)

    def to_piper_params(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"text"})


class PiperTextToSpeechRequest(PiperTextToSpeechConfig):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PiperTextToSpeechResponse(BaseModel):
    audio: bytes
    sample_rate: int
    content_type: str
