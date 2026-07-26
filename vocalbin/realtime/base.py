from dataclasses import dataclass
from typing import Any, cast

from pydantic import TypeAdapter

from vocalbin.models import (
    RealtimeSessionType,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptionEvent,
    RealtimeTranslationClosed,
    RealtimeTranslationEvent,
)
from vocalbin.realtime.models import (
    RealtimeAudioAppend,
    RealtimeInputFinished,
    RealtimeMessageType,
    RealtimeTranscriptionAudioAppend,
    RealtimeTranscriptionAudioCommit,
    RealtimeTranscriptionServerEvent,
    RealtimeTranslationAudioAppend,
    RealtimeTranslationServerEvent,
    RealtimeTranslationSessionClose,
)


@dataclass(frozen=True)
class RealtimeSessionSpec[
    EventT: (RealtimeTranscriptionEvent, RealtimeTranslationEvent)
]:
    """Everything that differs between the realtime session types.

    Transcription and translation speak the same websocket protocol but name
    their messages differently and end their streams on different events, so
    those differences are kept here as data instead of in client subclasses.
    """

    session_type: RealtimeSessionType
    audio_append: type[RealtimeAudioAppend]
    input_finished: type[RealtimeInputFinished]
    final_event: type[Any]
    final_event_requires_finished_input: bool
    event_types: frozenset[RealtimeMessageType]
    event_adapter: TypeAdapter[Any]

    def parse_event(self, payload: dict[str, Any]) -> EventT | None:
        # Sessions also emit lifecycle events this library does not expose.
        if payload.get("type") not in self.event_types:
            return None
        return cast(EventT, self.event_adapter.validate_python(payload).to_event())

    def is_final_event(self, event: EventT, *, input_finished: bool) -> bool:
        if not isinstance(event, self.final_event):
            return False
        return input_finished or not self.final_event_requires_finished_input


TRANSCRIPTION_SPEC = RealtimeSessionSpec[RealtimeTranscriptionEvent](
    session_type=RealtimeSessionType.TRANSCRIPTION,
    audio_append=RealtimeTranscriptionAudioAppend,
    input_finished=RealtimeTranscriptionAudioCommit,
    final_event=RealtimeTranscriptCompleted,
    # Transcripts complete per utterance, so only the one that follows the
    # final commit ends the stream.
    final_event_requires_finished_input=True,
    event_types=frozenset(
        {
            RealtimeMessageType.TRANSCRIPT_DELTA,
            RealtimeMessageType.TRANSCRIPT_COMPLETED,
            RealtimeMessageType.SPEECH_STARTED,
            RealtimeMessageType.SPEECH_STOPPED,
            RealtimeMessageType.ERROR,
        }
    ),
    event_adapter=TypeAdapter(RealtimeTranscriptionServerEvent),
)

TRANSLATION_SPEC = RealtimeSessionSpec[RealtimeTranslationEvent](
    session_type=RealtimeSessionType.TRANSLATION,
    audio_append=RealtimeTranslationAudioAppend,
    input_finished=RealtimeTranslationSessionClose,
    final_event=RealtimeTranslationClosed,
    # The service closes a session once, so the event always ends the stream.
    final_event_requires_finished_input=False,
    event_types=frozenset(
        {
            RealtimeMessageType.SOURCE_TRANSCRIPT_DELTA,
            RealtimeMessageType.TRANSLATION_TRANSCRIPT_DELTA,
            RealtimeMessageType.TRANSLATION_AUDIO_DELTA,
            RealtimeMessageType.TRANSLATION_CLOSED,
            RealtimeMessageType.ERROR,
        }
    ),
    event_adapter=TypeAdapter(RealtimeTranslationServerEvent),
)
