from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextToSpeechModel(StrEnum):
    GPT_4O_MINI_TTS = "gpt-4o-mini-tts"
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


class TextToSpeechVoice(StrEnum):
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    FABLE = "fable"
    NOVA = "nova"
    ONYX = "onyx"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"
    MARIN = "marin"
    CEDAR = "cedar"


class TextToSpeechFormat(StrEnum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class TextToSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: TextToSpeechModel | str = TextToSpeechModel.GPT_4O_MINI_TTS
    voice: TextToSpeechVoice | str = TextToSpeechVoice.MARIN
    response_format: TextToSpeechFormat = TextToSpeechFormat.MP3
    instructions: str | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)

    @model_validator(mode="after")
    def validate_model_capabilities(self) -> Self:
        legacy_models = {TextToSpeechModel.TTS_1, TextToSpeechModel.TTS_1_HD}
        legacy_voices = {
            TextToSpeechVoice.ALLOY,
            TextToSpeechVoice.ASH,
            TextToSpeechVoice.CORAL,
            TextToSpeechVoice.ECHO,
            TextToSpeechVoice.FABLE,
            TextToSpeechVoice.ONYX,
            TextToSpeechVoice.NOVA,
            TextToSpeechVoice.SAGE,
            TextToSpeechVoice.SHIMMER,
        }
        if self.model in legacy_models and self.voice not in legacy_voices:
            raise ValueError(f"{self.model} does not support voice '{self.voice}'")
        if self.model in legacy_models and self.instructions is not None:
            raise ValueError(f"{self.model} does not support instructions")
        return self


class TextToSpeechResponse(BaseModel):
    audio: bytes
    model: TextToSpeechModel | str
    voice: TextToSpeechVoice | str
    response_format: TextToSpeechFormat
    content_type: str
