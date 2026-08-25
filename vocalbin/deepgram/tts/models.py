from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vocalbin.deepgram.generated import AuraVoice

# Deepgram addresses a voice by picking the model that carries it.
TextToSpeechModel = AuraVoice


class AudioEncoding(StrEnum):
    LINEAR16 = "linear16"
    MULAW = "mulaw"
    ALAW = "alaw"
    MP3 = "mp3"
    OPUS = "opus"
    FLAC = "flac"
    AAC = "aac"


class AudioContainer(StrEnum):
    NONE = "none"
    WAV = "wav"
    OGG = "ogg"


_BIT_RATE_ENCODINGS = {AudioEncoding.MP3, AudioEncoding.AAC, AudioEncoding.OPUS}
_WAV_ENCODINGS = {AudioEncoding.LINEAR16, AudioEncoding.MULAW, AudioEncoding.ALAW}

CONTENT_TYPES = {
    AudioEncoding.LINEAR16: "audio/pcm",
    AudioEncoding.MULAW: "audio/basic",
    AudioEncoding.ALAW: "audio/alaw",
    AudioEncoding.MP3: "audio/mpeg",
    AudioEncoding.OPUS: "audio/ogg",
    AudioEncoding.FLAC: "audio/flac",
    AudioEncoding.AAC: "audio/aac",
}


class TextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: TextToSpeechModel | str = TextToSpeechModel.AURA_2_THALIA_EN
    encoding: AudioEncoding = AudioEncoding.LINEAR16
    container: AudioContainer | None = None
    sample_rate: int | None = Field(default=24000, gt=0)
    bit_rate: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def encoding_must_support_options(self) -> Self:
        if self.bit_rate is not None and self.encoding not in _BIT_RATE_ENCODINGS:
            raise ValueError(f"bit_rate is not supported by encoding {self.encoding}")
        if self.container == AudioContainer.WAV and self.encoding not in _WAV_ENCODINGS:
            raise ValueError(
                f"container 'wav' is not supported by encoding {self.encoding}"
            )
        if self.container == AudioContainer.OGG and self.encoding != AudioEncoding.OPUS:
            raise ValueError("container 'ogg' requires encoding 'opus'")
        return self

    @property
    def content_type(self) -> str:
        if self.container == AudioContainer.WAV:
            return "audio/wav"
        if self.container == AudioContainer.OGG:
            return "audio/ogg"
        return CONTENT_TYPES[AudioEncoding(self.encoding)]


class TextToSpeechResponse(BaseModel):
    audio: bytes
    model: TextToSpeechModel | str
    encoding: AudioEncoding
    container: AudioContainer | None
    sample_rate: int | None
    content_type: str
