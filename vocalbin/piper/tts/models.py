from pydantic import BaseModel, ConfigDict, Field


class TextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_id: int | None = Field(default=None, ge=0)
    length_scale: float | None = Field(default=None, gt=0)
    noise_scale: float | None = Field(default=None, ge=0)
    noise_w_scale: float | None = Field(default=None, ge=0)


class TextToSpeechResponse(BaseModel):
    audio: bytes
    sample_rate: int
    content_type: str
