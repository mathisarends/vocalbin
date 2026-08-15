from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CartesiaTextToSpeechModel(StrEnum):
    SONIC_3_5 = "sonic-3.5"
    SONIC_3 = "sonic-3"
    SONIC_3_5_2026_05_04 = "sonic-3.5-2026-05-04"
    SONIC_3_2026_01_12 = "sonic-3-2026-01-12"
    SONIC_3_2025_10_27 = "sonic-3-2025-10-27"
    SONIC_LATEST = "sonic-latest"


class CartesiaAudioEncoding(StrEnum):
    PCM_F32LE = "pcm_f32le"
    PCM_S16LE = "pcm_s16le"
    PCM_MULAW = "pcm_mulaw"
    PCM_ALAW = "pcm_alaw"


type CartesiaSampleRate = Literal[8000, 16000, 22050, 24000, 44100, 48000]
type CartesiaBitRate = Literal[32000, 64000, 96000, 128000, 192000]


class CartesiaRawOutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["raw"] = "raw"
    encoding: CartesiaAudioEncoding = CartesiaAudioEncoding.PCM_S16LE
    sample_rate: CartesiaSampleRate = 24000


class CartesiaWavOutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["wav"] = "wav"
    encoding: CartesiaAudioEncoding = CartesiaAudioEncoding.PCM_S16LE
    sample_rate: CartesiaSampleRate = 44100


class CartesiaMp3OutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["mp3"] = "mp3"
    sample_rate: CartesiaSampleRate = 44100
    bit_rate: CartesiaBitRate = 128000


type CartesiaOutputFormat = Annotated[
    CartesiaRawOutputFormat | CartesiaWavOutputFormat | CartesiaMp3OutputFormat,
    Field(discriminator="container"),
]


class CartesiaGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion: str | None = None
    speed: float | None = Field(default=None, ge=0.6, le=1.5)
    volume: float | None = Field(default=None, ge=0.5, le=2.0)

    @field_validator("emotion")
    @classmethod
    def emotion_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("emotion must not be blank")
        return value

    def to_cartesia_params(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, mode="json")


class CartesiaTextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice_id: str = Field(min_length=1)
    model: CartesiaTextToSpeechModel | str = CartesiaTextToSpeechModel.SONIC_3_5
    output_format: CartesiaOutputFormat = Field(
        default_factory=CartesiaRawOutputFormat,
        discriminator="container",
    )
    language: str | None = None
    generation_config: CartesiaGenerationConfig | None = None
    pronunciation_dict_id: str | None = None
    max_buffer_delay_ms: int | None = Field(default=None, ge=0)
    timeout: float | None = Field(default=None, gt=0)

    @field_validator("voice_id", "pronunciation_dict_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("identifier must not be blank")
        return value

    @field_validator("language")
    @classmethod
    def language_must_be_iso_639_1(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 2 or not normalized.isalpha() or not normalized.isascii():
            raise ValueError("language must be an ISO-639-1 code like 'de' or 'en'")
        return normalized

    def to_cartesia_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model_id": (
                self.model.value
                if isinstance(self.model, CartesiaTextToSpeechModel)
                else self.model
            ),
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": self.output_format.model_dump(mode="json"),
        }
        if self.language is not None:
            params["language"] = self.language
        if self.generation_config is not None:
            params["generation_config"] = self.generation_config.to_cartesia_params()
        if self.pronunciation_dict_id is not None:
            params["pronunciation_dict_id"] = self.pronunciation_dict_id
        return params


class CartesiaTextToSpeechResponse(BaseModel):
    audio: bytes
    model: CartesiaTextToSpeechModel | str
    voice_id: str
    output_format: CartesiaOutputFormat
    content_type: str
