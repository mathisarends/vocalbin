import base64
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from vocalbin.models import (
    RealtimeError,
    RealtimeErrorDetails,
    RealtimeNoiseReduction,
    RealtimeSourceTranscriptDelta,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionDelay,
    RealtimeTranscriptionModel,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationClosed,
    RealtimeTranslationLanguage,
    RealtimeTranslationTranscriptDelta,
)


class RealtimeMessageType(StrEnum):
    PCM_AUDIO = "audio/pcm"
    TRANSCRIPTION_SESSION = "transcription"
    SESSION_UPDATE = "session.update"
    TRANSCRIPTION_AUDIO_APPEND = "input_audio_buffer.append"
    TRANSCRIPTION_AUDIO_COMMIT = "input_audio_buffer.commit"
    TRANSLATION_AUDIO_APPEND = "session.input_audio_buffer.append"
    TRANSLATION_SESSION_CLOSE = "session.close"
    TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
    TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
    SPEECH_STARTED = "input_audio_buffer.speech_started"
    SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    SOURCE_TRANSCRIPT_DELTA = "session.input_transcript.delta"
    TRANSLATION_TRANSCRIPT_DELTA = "session.output_transcript.delta"
    TRANSLATION_AUDIO_DELTA = "session.output_audio.delta"
    TRANSLATION_CLOSED = "session.closed"
    ERROR = "error"


class RealtimePcmFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.PCM_AUDIO] = RealtimeMessageType.PCM_AUDIO
    rate: Literal[24000] = 24000


class RealtimeNoiseReductionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RealtimeNoiseReduction


class RealtimeTranscriptionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: RealtimeTranscriptionModel
    delay: RealtimeTranscriptionDelay
    language: str | None = Field(default=None, exclude_if=lambda value: value is None)


class RealtimeTranscriptionAudioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: RealtimePcmFormat
    transcription: RealtimeTranscriptionSettings
    turn_detection: None = None
    noise_reduction: RealtimeNoiseReductionConfig | None


class RealtimeTranscriptionAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: RealtimeTranscriptionAudioInput


class RealtimeTranscriptionSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.TRANSCRIPTION_SESSION] = (
        RealtimeMessageType.TRANSCRIPTION_SESSION
    )
    audio: RealtimeTranscriptionAudio
    include: list[Literal["item.input_audio_transcription.logprobs"]] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class RealtimeTranscriptionSessionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.SESSION_UPDATE] = (
        RealtimeMessageType.SESSION_UPDATE
    )
    session: RealtimeTranscriptionSession


class RealtimeTranslationTranscriptionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: RealtimeTranscriptionModel = RealtimeTranscriptionModel.GPT_REALTIME_WHISPER


class RealtimeTranslationInputAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcription: RealtimeTranslationTranscriptionSettings | None
    noise_reduction: RealtimeNoiseReductionConfig | None


class RealtimeTranslationOutputAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: RealtimeTranslationLanguage


class RealtimeTranslationAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: RealtimeTranslationInputAudio
    output: RealtimeTranslationOutputAudio


class RealtimeTranslationSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: RealtimeTranslationAudio


class RealtimeTranslationSessionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.SESSION_UPDATE] = (
        RealtimeMessageType.SESSION_UPDATE
    )
    session: RealtimeTranslationSession


class RealtimeTranscriptionAudioAppend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.TRANSCRIPTION_AUDIO_APPEND] = (
        RealtimeMessageType.TRANSCRIPTION_AUDIO_APPEND
    )
    audio: str

    @classmethod
    def from_audio(cls, audio: bytes) -> "RealtimeTranscriptionAudioAppend":
        return cls(audio=base64.b64encode(audio).decode("ascii"))


class RealtimeTranscriptionAudioCommit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.TRANSCRIPTION_AUDIO_COMMIT] = (
        RealtimeMessageType.TRANSCRIPTION_AUDIO_COMMIT
    )


class RealtimeTranslationAudioAppend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.TRANSLATION_AUDIO_APPEND] = (
        RealtimeMessageType.TRANSLATION_AUDIO_APPEND
    )
    audio: str

    @classmethod
    def from_audio(cls, audio: bytes) -> "RealtimeTranslationAudioAppend":
        return cls(audio=base64.b64encode(audio).decode("ascii"))


class RealtimeTranslationSessionClose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[RealtimeMessageType.TRANSLATION_SESSION_CLOSE] = (
        RealtimeMessageType.TRANSLATION_SESSION_CLOSE
    )


type RealtimeSessionUpdate = (
    RealtimeTranscriptionSessionUpdate | RealtimeTranslationSessionUpdate
)
type RealtimeAudioAppend = (
    RealtimeTranscriptionAudioAppend | RealtimeTranslationAudioAppend
)
type RealtimeInputFinished = (
    RealtimeTranscriptionAudioCommit | RealtimeTranslationSessionClose
)
type RealtimeClientMessage = (
    RealtimeSessionUpdate | RealtimeAudioAppend | RealtimeInputFinished
)


class RealtimeServerEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RealtimeErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    message: str
    code: str | None = None
    event_id: str | None = None
    param: str | None = None

    def to_error_details(self) -> RealtimeErrorDetails:
        return RealtimeErrorDetails(
            type=self.type,
            message=self.message,
            code=self.code,
            event_id=self.event_id,
            param=self.param,
        )


class RealtimeErrorEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.ERROR]
    error: RealtimeErrorPayload

    def to_event(self) -> RealtimeError:
        return RealtimeError(error=self.error.to_error_details())


class RealtimeTranscriptDeltaEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.TRANSCRIPT_DELTA]
    delta: str
    item_id: str
    event_id: str | None = None
    logprobs: list[dict[str, Any]] | None = None

    def to_event(self) -> RealtimeTranscriptDelta:
        return RealtimeTranscriptDelta(
            delta=self.delta,
            item_id=self.item_id,
            event_id=self.event_id,
            logprobs=self.logprobs,
        )


class RealtimeTranscriptCompletedEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.TRANSCRIPT_COMPLETED]
    transcript: str
    item_id: str
    event_id: str | None = None
    logprobs: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None

    def to_event(self) -> RealtimeTranscriptCompleted:
        return RealtimeTranscriptCompleted(
            transcript=self.transcript,
            item_id=self.item_id,
            event_id=self.event_id,
            logprobs=self.logprobs,
            usage=self.usage,
        )


class RealtimeSpeechStartedEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.SPEECH_STARTED]
    item_id: str
    audio_start_ms: int

    def to_event(self) -> RealtimeSpeechStarted:
        return RealtimeSpeechStarted(
            item_id=self.item_id,
            audio_start_ms=self.audio_start_ms,
        )


class RealtimeSpeechStoppedEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.SPEECH_STOPPED]
    item_id: str
    audio_end_ms: int

    def to_event(self) -> RealtimeSpeechStopped:
        return RealtimeSpeechStopped(
            item_id=self.item_id,
            audio_end_ms=self.audio_end_ms,
        )


class RealtimeSourceTranscriptDeltaEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.SOURCE_TRANSCRIPT_DELTA]
    delta: str
    elapsed_ms: int | None = None
    event_id: str | None = None

    def to_event(self) -> RealtimeSourceTranscriptDelta:
        return RealtimeSourceTranscriptDelta(
            delta=self.delta,
            elapsed_ms=self.elapsed_ms,
            event_id=self.event_id,
        )


class RealtimeTranslationTranscriptDeltaEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.TRANSLATION_TRANSCRIPT_DELTA]
    delta: str
    elapsed_ms: int | None = None
    event_id: str | None = None

    def to_event(self) -> RealtimeTranslationTranscriptDelta:
        return RealtimeTranslationTranscriptDelta(
            delta=self.delta,
            elapsed_ms=self.elapsed_ms,
            event_id=self.event_id,
        )


class RealtimeTranslationAudioDeltaEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.TRANSLATION_AUDIO_DELTA]
    delta: str
    elapsed_ms: int | None = None
    sample_rate: int = 24000
    channels: int = 1
    format: Literal["pcm16"] = "pcm16"
    event_id: str | None = None

    def to_event(self) -> RealtimeTranslationAudioDelta:
        return RealtimeTranslationAudioDelta(
            audio=base64.b64decode(self.delta, validate=True),
            elapsed_ms=self.elapsed_ms,
            sample_rate=self.sample_rate,
            channels=self.channels,
            format=self.format,
            event_id=self.event_id,
        )


class RealtimeTranslationClosedEvent(RealtimeServerEvent):
    type: Literal[RealtimeMessageType.TRANSLATION_CLOSED]
    event_id: str | None = None

    def to_event(self) -> RealtimeTranslationClosed:
        return RealtimeTranslationClosed(event_id=self.event_id)


type RealtimeTranscriptionServerEvent = Annotated[
    RealtimeTranscriptDeltaEvent
    | RealtimeTranscriptCompletedEvent
    | RealtimeSpeechStartedEvent
    | RealtimeSpeechStoppedEvent
    | RealtimeErrorEvent,
    Field(discriminator="type"),
]

type RealtimeTranslationServerEvent = Annotated[
    RealtimeSourceTranscriptDeltaEvent
    | RealtimeTranslationTranscriptDeltaEvent
    | RealtimeTranslationAudioDeltaEvent
    | RealtimeTranslationClosedEvent
    | RealtimeErrorEvent,
    Field(discriminator="type"),
]

TRANSCRIPTION_EVENT_TYPES = {
    RealtimeMessageType.TRANSCRIPT_DELTA,
    RealtimeMessageType.TRANSCRIPT_COMPLETED,
    RealtimeMessageType.SPEECH_STARTED,
    RealtimeMessageType.SPEECH_STOPPED,
    RealtimeMessageType.ERROR,
}
TRANSLATION_EVENT_TYPES = {
    RealtimeMessageType.SOURCE_TRANSCRIPT_DELTA,
    RealtimeMessageType.TRANSLATION_TRANSCRIPT_DELTA,
    RealtimeMessageType.TRANSLATION_AUDIO_DELTA,
    RealtimeMessageType.TRANSLATION_CLOSED,
    RealtimeMessageType.ERROR,
}

transcription_event_adapter = TypeAdapter(RealtimeTranscriptionServerEvent)
translation_event_adapter = TypeAdapter(RealtimeTranslationServerEvent)
