from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class TextToSpeechModel(StrEnum):
    AURA_2_AMALTHEA_EN = "aura-2-amalthea-en"
    AURA_2_ANDROMEDA_EN = "aura-2-andromeda-en"
    AURA_2_APOLLO_EN = "aura-2-apollo-en"
    AURA_2_ARCAS_EN = "aura-2-arcas-en"
    AURA_2_ARIES_EN = "aura-2-aries-en"
    AURA_2_ASTERIA_EN = "aura-2-asteria-en"
    AURA_2_ATHENA_EN = "aura-2-athena-en"
    AURA_2_ATLAS_EN = "aura-2-atlas-en"
    AURA_2_AURORA_EN = "aura-2-aurora-en"
    AURA_2_CALLISTA_EN = "aura-2-callista-en"
    AURA_2_CORA_EN = "aura-2-cora-en"
    AURA_2_CORDELIA_EN = "aura-2-cordelia-en"
    AURA_2_DELIA_EN = "aura-2-delia-en"
    AURA_2_DRACO_EN = "aura-2-draco-en"
    AURA_2_ELECTRA_EN = "aura-2-electra-en"
    AURA_2_HARMONIA_EN = "aura-2-harmonia-en"
    AURA_2_HELENA_EN = "aura-2-helena-en"
    AURA_2_HERA_EN = "aura-2-hera-en"
    AURA_2_HERMES_EN = "aura-2-hermes-en"
    AURA_2_HYPERION_EN = "aura-2-hyperion-en"
    AURA_2_IRIS_EN = "aura-2-iris-en"
    AURA_2_JANUS_EN = "aura-2-janus-en"
    AURA_2_JUNO_EN = "aura-2-juno-en"
    AURA_2_JUPITER_EN = "aura-2-jupiter-en"
    AURA_2_LUNA_EN = "aura-2-luna-en"
    AURA_2_MARS_EN = "aura-2-mars-en"
    AURA_2_MINERVA_EN = "aura-2-minerva-en"
    AURA_2_NEPTUNE_EN = "aura-2-neptune-en"
    AURA_2_ODYSSEUS_EN = "aura-2-odysseus-en"
    AURA_2_OPHELIA_EN = "aura-2-ophelia-en"
    AURA_2_ORION_EN = "aura-2-orion-en"
    AURA_2_ORPHEUS_EN = "aura-2-orpheus-en"
    AURA_2_PANDORA_EN = "aura-2-pandora-en"
    AURA_2_PHOEBE_EN = "aura-2-phoebe-en"
    AURA_2_PLUTO_EN = "aura-2-pluto-en"
    AURA_2_SATURN_EN = "aura-2-saturn-en"
    AURA_2_SELENE_EN = "aura-2-selene-en"
    AURA_2_THALIA_EN = "aura-2-thalia-en"
    AURA_2_THEIA_EN = "aura-2-theia-en"
    AURA_2_VESTA_EN = "aura-2-vesta-en"
    AURA_2_ZEUS_EN = "aura-2-zeus-en"
    AURA_2_ALVARO_ES = "aura-2-alvaro-es"
    AURA_2_AQUILA_ES = "aura-2-aquila-es"
    AURA_2_CARINA_ES = "aura-2-carina-es"
    AURA_2_CELESTE_ES = "aura-2-celeste-es"
    AURA_2_DIANA_ES = "aura-2-diana-es"
    AURA_2_ESTRELLA_ES = "aura-2-estrella-es"
    AURA_2_JAVIER_ES = "aura-2-javier-es"
    AURA_2_NESTOR_ES = "aura-2-nestor-es"
    AURA_2_SELENA_ES = "aura-2-selena-es"
    AURA_2_SIRIO_ES = "aura-2-sirio-es"


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

_SAMPLE_RATES = {
    AudioEncoding.LINEAR16: {8000, 16000, 24000, 32000, 48000},
    AudioEncoding.MULAW: {8000, 16000},
    AudioEncoding.ALAW: {8000, 16000},
    AudioEncoding.FLAC: {8000, 16000, 22050, 32000, 48000},
}
_DEFAULT_SAMPLE_RATES = {
    AudioEncoding.LINEAR16: 24000,
    AudioEncoding.MULAW: 8000,
    AudioEncoding.ALAW: 8000,
    AudioEncoding.FLAC: 48000,
}

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
    sample_rate: int | None = Field(default=None, gt=0, validate_default=True)
    bit_rate: int | None = Field(default=None, gt=0)

    @field_validator("sample_rate")
    @classmethod
    def default_sample_rate(cls, value: int | None, info: ValidationInfo) -> int | None:
        if value is not None:
            return value
        return _DEFAULT_SAMPLE_RATES.get(info.data.get("encoding"))

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
        if self.container is not None and self.encoding in {
            AudioEncoding.MP3,
            AudioEncoding.FLAC,
            AudioEncoding.AAC,
        }:
            raise ValueError(f"container is not supported by encoding {self.encoding}")
        if (
            self.container == AudioContainer.NONE
            and self.encoding == AudioEncoding.OPUS
        ):
            raise ValueError("encoding opus requires container ogg")
        if self.sample_rate is not None:
            rates = _SAMPLE_RATES.get(self.encoding, set())
            if self.sample_rate not in rates:
                raise ValueError(
                    f"sample_rate is not supported by encoding {self.encoding}"
                )
        if self.bit_rate is not None:
            if self.encoding == AudioEncoding.MP3 and self.bit_rate not in {
                32000,
                48000,
            }:
                raise ValueError("mp3 bit_rate must be 32000 or 48000")
            if (
                self.encoding == AudioEncoding.OPUS
                and not 4000 <= self.bit_rate <= 650000
            ):
                raise ValueError("opus bit_rate must be between 4000 and 650000")
            if (
                self.encoding == AudioEncoding.AAC
                and not 4000 <= self.bit_rate <= 192000
            ):
                raise ValueError("aac bit_rate must be between 4000 and 192000")
        return self

    @property
    def effective_container(self) -> AudioContainer | None:
        if self.container is not None:
            return self.container
        if self.encoding in _WAV_ENCODINGS:
            return AudioContainer.WAV
        if self.encoding == AudioEncoding.OPUS:
            return AudioContainer.OGG
        return None

    @property
    def effective_sample_rate(self) -> int | None:
        if self.sample_rate is not None:
            return self.sample_rate
        return {
            AudioEncoding.MP3: 22050,
            AudioEncoding.AAC: 22050,
            AudioEncoding.OPUS: 48000,
        }.get(self.encoding)

    @property
    def content_type(self) -> str:
        if self.effective_container == AudioContainer.WAV:
            return "audio/wav"
        if self.effective_container == AudioContainer.OGG:
            return "audio/ogg"
        return CONTENT_TYPES[AudioEncoding(self.encoding)]


class TextToSpeechResponse(BaseModel):
    audio: bytes
    model: TextToSpeechModel | str
    encoding: AudioEncoding
    container: AudioContainer | None
    sample_rate: int | None
    content_type: str
