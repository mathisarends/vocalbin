import base64
from abc import abstractmethod
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from vocalbin.openai.realtime.models import (
    RealtimeError,
    RealtimeErrorDetails,
    RealtimeLogprob,
    RealtimeNoiseReduction,
    RealtimeSourceTranscriptDelta,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionDelay,
    RealtimeTranscriptionEvent,
    RealtimeTranscriptionModel,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationClosed,
    RealtimeTranslationConfig,
    RealtimeTranslationEvent,
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


class RealtimeClientMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RealtimePcmFormat(RealtimeClientMessage):
    type: Literal[RealtimeMessageType.PCM_AUDIO] = RealtimeMessageType.PCM_AUDIO
    rate: Literal[24000] = 24000


class RealtimeNoiseReductionConfig(RealtimeClientMessage):
    type: RealtimeNoiseReduction

    @classmethod
    def from_setting(cls, setting: RealtimeNoiseReduction | None) -> Self | None:
        return None if setting is None else cls(type=setting)


class RealtimeTranscriptionSettings(RealtimeClientMessage):
    model: RealtimeTranscriptionModel | str
    delay: RealtimeTranscriptionDelay
    language: str | None = Field(default=None, exclude_if=lambda value: value is None)


class RealtimeTranscriptionAudioInput(RealtimeClientMessage):
    format: RealtimePcmFormat
    transcription: RealtimeTranscriptionSettings
    turn_detection: None = None
    noise_reduction: RealtimeNoiseReductionConfig | None


class RealtimeTranscriptionAudio(RealtimeClientMessage):
    input: RealtimeTranscriptionAudioInput


class RealtimeTranscriptionSession(RealtimeClientMessage):
    type: Literal[RealtimeMessageType.TRANSCRIPTION_SESSION] = (
        RealtimeMessageType.TRANSCRIPTION_SESSION
    )
    audio: RealtimeTranscriptionAudio
    include: list[Literal["item.input_audio_transcription.logprobs"]] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class RealtimeTranslationTranscriptionSettings(RealtimeClientMessage):
    model: RealtimeTranscriptionModel | str = (
        RealtimeTranscriptionModel.GPT_REALTIME_WHISPER
    )


class RealtimeTranslationInputAudio(RealtimeClientMessage):
    transcription: RealtimeTranslationTranscriptionSettings | None
    noise_reduction: RealtimeNoiseReductionConfig | None


class RealtimeTranslationOutputAudio(RealtimeClientMessage):
    language: RealtimeTranslationLanguage


class RealtimeTranslationAudio(RealtimeClientMessage):
    input: RealtimeTranslationInputAudio
    output: RealtimeTranslationOutputAudio


class RealtimeTranslationSession(RealtimeClientMessage):
    audio: RealtimeTranslationAudio


class RealtimeSessionUpdate(RealtimeClientMessage):
    type: Literal[RealtimeMessageType.SESSION_UPDATE] = (
        RealtimeMessageType.SESSION_UPDATE
    )


class RealtimeTranscriptionSessionUpdate(RealtimeSessionUpdate):
    session: RealtimeTranscriptionSession

    @classmethod
    def from_config(cls, config: RealtimeTranscriptionConfig) -> Self:
        return cls(
            session=RealtimeTranscriptionSession(
                audio=RealtimeTranscriptionAudio(
                    input=RealtimeTranscriptionAudioInput(
                        format=RealtimePcmFormat(),
                        transcription=RealtimeTranscriptionSettings(
                            model=config.model,
                            delay=config.delay,
                            language=config.language,
                        ),
                        noise_reduction=RealtimeNoiseReductionConfig.from_setting(
                            config.noise_reduction
                        ),
                    )
                ),
                include=(
                    ["item.input_audio_transcription.logprobs"]
                    if config.include_logprobs
                    else None
                ),
            )
        )


class RealtimeTranslationSessionUpdate(RealtimeSessionUpdate):
    session: RealtimeTranslationSession

    @classmethod
    def from_config(cls, config: RealtimeTranslationConfig) -> Self:
        return cls(
            session=RealtimeTranslationSession(
                audio=RealtimeTranslationAudio(
                    input=RealtimeTranslationInputAudio(
                        transcription=(
                            RealtimeTranslationTranscriptionSettings()
                            if config.include_source_transcript
                            else None
                        ),
                        noise_reduction=RealtimeNoiseReductionConfig.from_setting(
                            config.noise_reduction
                        ),
                    ),
                    output=RealtimeTranslationOutputAudio(
                        language=config.target_language
                    ),
                )
            )
        )


class RealtimeAudioAppend(RealtimeClientMessage):
    audio: str

    @classmethod
    def from_audio(cls, audio: bytes) -> Self:
        return cls(audio=base64.b64encode(audio).decode("ascii"))


class RealtimeTranscriptionAudioAppend(RealtimeAudioAppend):
    type: Literal[RealtimeMessageType.TRANSCRIPTION_AUDIO_APPEND] = (
        RealtimeMessageType.TRANSCRIPTION_AUDIO_APPEND
    )


class RealtimeTranslationAudioAppend(RealtimeAudioAppend):
    type: Literal[RealtimeMessageType.TRANSLATION_AUDIO_APPEND] = (
        RealtimeMessageType.TRANSLATION_AUDIO_APPEND
    )


class RealtimeInputFinished(RealtimeClientMessage):
    """Announces that no further audio will follow."""


class RealtimeTranscriptionAudioCommit(RealtimeInputFinished):
    type: Literal[RealtimeMessageType.TRANSCRIPTION_AUDIO_COMMIT] = (
        RealtimeMessageType.TRANSCRIPTION_AUDIO_COMMIT
    )


class RealtimeTranslationSessionClose(RealtimeInputFinished):
    type: Literal[RealtimeMessageType.TRANSLATION_SESSION_CLOSE] = (
        RealtimeMessageType.TRANSLATION_SESSION_CLOSE
    )


class RealtimeServerEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @abstractmethod
    def to_event(self) -> RealtimeTranscriptionEvent | RealtimeTranslationEvent: ...


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
    logprobs: list[RealtimeLogprob] | None = None

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
    logprobs: list[RealtimeLogprob] | None = None
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
