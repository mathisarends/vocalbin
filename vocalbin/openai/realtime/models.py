from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

type RealtimeTranscriptionModel = Literal[
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-transcribe",
    "gpt-4o-transcribe-diarize",
    "gpt-realtime-whisper",
]
type RealtimeTranslationModel = Literal["gpt-realtime-translate"]


class RealtimeSessionType(StrEnum):
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"


class RealtimeTranscriptionDelay(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class RealtimeNoiseReduction(StrEnum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class RealtimeTranslationLanguage(StrEnum):
    CHINESE = "zh"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    HINDI = "hi"
    INDONESIAN = "id"
    ITALIAN = "it"
    JAPANESE = "ja"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    SPANISH = "es"
    VIETNAMESE = "vi"


class SemanticVadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["semantic_vad"] = "semantic_vad"
    eagerness: Literal["low", "medium", "high"] = "medium"


class RealtimeTranscriptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: RealtimeTranscriptionModel = "gpt-realtime-whisper"
    language: str | None = None
    delay: RealtimeTranscriptionDelay = RealtimeTranscriptionDelay.MEDIUM
    noise_reduction: RealtimeNoiseReduction | None = RealtimeNoiseReduction.FAR_FIELD
    turn_detection: SemanticVadConfig | None = None
    include_logprobs: bool = False

    @model_validator(mode="after")
    def validate_turn_detection_support(self) -> "RealtimeTranscriptionConfig":
        if self.model == "gpt-realtime-whisper" and self.turn_detection is not None:
            raise ValueError(
                "gpt-realtime-whisper does not support turn detection; "
                "use gpt-4o-transcribe or set turn_detection=None"
            )
        return self

    @field_validator("language")
    @classmethod
    def language_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("language must not be blank")
        return normalized


class RealtimeTranslationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: RealtimeTranslationModel = "gpt-realtime-translate"
    target_language: RealtimeTranslationLanguage
    noise_reduction: RealtimeNoiseReduction | None = RealtimeNoiseReduction.FAR_FIELD
    include_source_transcript: bool = True


class RealtimeEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RealtimeLogprob(BaseModel):
    # Scored alternatives come straight from the API, which may add fields.
    model_config = ConfigDict(frozen=True, extra="ignore")

    token: str
    logprob: float
    bytes: list[int] | None = None


class RealtimeErrorDetails(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    message: str
    code: str | None = None
    event_id: str | None = None
    param: str | None = None

    def __str__(self) -> str:
        return f"[{self.type}] {self.message}"


class RealtimeSessionConnected(RealtimeEvent):
    pass


class RealtimeError(RealtimeEvent):
    error: RealtimeErrorDetails


class RealtimeTranscriptDelta(RealtimeEvent):
    delta: str
    item_id: str
    event_id: str | None = None
    logprobs: list[RealtimeLogprob] | None = None


class RealtimeTranscriptCompleted(RealtimeEvent):
    transcript: str
    item_id: str
    event_id: str | None = None
    logprobs: list[RealtimeLogprob] | None = None
    usage: dict[str, Any] | None = None


class RealtimeSpeechStarted(RealtimeEvent):
    item_id: str
    audio_start_ms: int


class RealtimeSpeechStopped(RealtimeEvent):
    item_id: str
    audio_end_ms: int


class RealtimeSourceTranscriptDelta(RealtimeEvent):
    delta: str
    elapsed_ms: int | None = None
    event_id: str | None = None


class RealtimeTranslationTranscriptDelta(RealtimeEvent):
    delta: str
    elapsed_ms: int | None = None
    event_id: str | None = None


class RealtimeTranslationAudioDelta(RealtimeEvent):
    audio: bytes
    elapsed_ms: int | None = None
    sample_rate: int = 24000
    channels: int = 1
    format: Literal["pcm16"] = "pcm16"
    event_id: str | None = None


class RealtimeTranslationClosed(RealtimeEvent):
    event_id: str | None = None


type RealtimeTranscriptionEvent = (
    RealtimeSessionConnected
    | RealtimeTranscriptDelta
    | RealtimeTranscriptCompleted
    | RealtimeSpeechStarted
    | RealtimeSpeechStopped
    | RealtimeError
)

type RealtimeTranslationEvent = (
    RealtimeSessionConnected
    | RealtimeSourceTranscriptDelta
    | RealtimeTranslationTranscriptDelta
    | RealtimeTranslationAudioDelta
    | RealtimeTranslationClosed
    | RealtimeError
)
