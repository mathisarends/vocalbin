from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextToSpeechModel(StrEnum):
    SONIC_3_5 = "sonic-3.5"
    SONIC_3 = "sonic-3"
    SONIC_3_5_2026_05_04 = "sonic-3.5-2026-05-04"
    SONIC_3_2026_01_12 = "sonic-3-2026-01-12"
    SONIC_3_2025_10_27 = "sonic-3-2025-10-27"
    SONIC_LATEST = "sonic-latest"


class AudioEncoding(StrEnum):
    PCM_F32LE = "pcm_f32le"
    PCM_S16LE = "pcm_s16le"
    PCM_MULAW = "pcm_mulaw"
    PCM_ALAW = "pcm_alaw"


type SampleRate = Literal[8000, 16000, 22050, 24000, 44100, 48000]
type BitRate = Literal[32000, 64000, 96000, 128000, 192000]


class RawOutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["raw"] = "raw"
    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    sample_rate: SampleRate = 24000


class WavOutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["wav"] = "wav"
    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    sample_rate: SampleRate = 44100


class Mp3OutputFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["mp3"] = "mp3"
    sample_rate: SampleRate = 44100
    bit_rate: BitRate = 128000


type OutputFormat = Annotated[
    RawOutputFormat | WavOutputFormat | Mp3OutputFormat,
    Field(discriminator="container"),
]


class GenerationConfig(BaseModel):
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


class TextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice_id: str = Field(min_length=1)
    model: TextToSpeechModel | str = Field(
        default=TextToSpeechModel.SONIC_3_5,
        serialization_alias="model_id",
    )
    output_format: OutputFormat = Field(
        default_factory=RawOutputFormat,
        discriminator="container",
    )
    language: str | None = None
    generation_config: GenerationConfig | None = None
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


class TextToSpeechResponse(BaseModel):
    audio: bytes
    model: TextToSpeechModel | str
    voice_id: str
    output_format: OutputFormat
    content_type: str
