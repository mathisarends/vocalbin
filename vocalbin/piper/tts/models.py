from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: int | None = Field(default=None, ge=0)
    length_scale: float | None = Field(default=None, gt=0)
    noise_scale: float | None = Field(default=None, ge=0)
    noise_w_scale: float | None = Field(default=None, ge=0)

    def to_piper_params(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TextToSpeechResponse(BaseModel):
    audio: bytes
    sample_rate: int
    content_type: str
