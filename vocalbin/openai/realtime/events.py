from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

type AudioFormat = Literal["pcm16"]


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Logprob(BaseModel):
    # Scored alternatives come straight from the API, which may add fields.
    model_config = ConfigDict(frozen=True, extra="ignore")

    token: str
    logprob: float
    bytes: list[int] | None = None


class ErrorDetails(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    message: str
    code: str | None = None
    event_id: str | None = None
    param: str | None = None

    def __str__(self) -> str:
        return f"[{self.type}] {self.message}"


class SessionConnected(Event):
    pass


class Error(Event):
    error: ErrorDetails


class TranscriptDelta(Event):
    delta: str
    item_id: str
    event_id: str | None = None
    logprobs: list[Logprob] | None = None


class TranscriptCompleted(Event):
    transcript: str
    item_id: str
    event_id: str | None = None
    logprobs: list[Logprob] | None = None
    usage: dict[str, Any] | None = None


class SpeechStarted(Event):
    item_id: str
    audio_start_ms: int


class SpeechStopped(Event):
    item_id: str
    audio_end_ms: int


class SourceTranscriptDelta(Event):
    delta: str
    elapsed_ms: int | None = None
    event_id: str | None = None


class TranslationTranscriptDelta(Event):
    delta: str
    elapsed_ms: int | None = None
    event_id: str | None = None


class TranslationAudioDelta(Event):
    audio: bytes
    elapsed_ms: int | None = None
    sample_rate: int = 24000
    channels: int = 1
    format: AudioFormat = "pcm16"
    event_id: str | None = None


class TranslationClosed(Event):
    event_id: str | None = None


type TranscriptionEvent = (
    SessionConnected
    | TranscriptDelta
    | TranscriptCompleted
    | SpeechStarted
    | SpeechStopped
    | Error
)

type TranslationEvent = (
    SessionConnected
    | SourceTranscriptDelta
    | TranslationTranscriptDelta
    | TranslationAudioDelta
    | TranslationClosed
    | Error
)


__all__ = [
    "AudioFormat",
    "Error",
    "ErrorDetails",
    "Event",
    "Logprob",
    "SessionConnected",
    "SourceTranscriptDelta",
    "SpeechStarted",
    "SpeechStopped",
    "TranscriptCompleted",
    "TranscriptDelta",
    "TranscriptionEvent",
    "TranslationAudioDelta",
    "TranslationClosed",
    "TranslationEvent",
    "TranslationTranscriptDelta",
]
