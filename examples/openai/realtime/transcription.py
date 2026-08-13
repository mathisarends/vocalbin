import asyncio

from examples.openai.realtime._terminal import RealtimeTerminal
from vocalbin.openai.realtime import (
    OpenAIRealtimeTranscriberBuilder,
    RealtimeError,
    RealtimeSessionConnected,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
)


async def main() -> None:
    partial_transcript = ""

    with RealtimeTerminal(
        title="Live transcription · German",
        fields=["transcript"],
    ) as terminal:
        transcriber = (
            OpenAIRealtimeTranscriberBuilder()
            .model("gpt-4o-transcribe")
            .language("de")
            .semantic_vad(eagerness="low")
            .build()
        )
        async with transcriber:
            async for event in transcriber.stream():
                match event:
                    case RealtimeSessionConnected():
                        terminal.update(status="Ready")
                    case RealtimeSpeechStarted():
                        terminal.update(status="Listening")
                    case RealtimeSpeechStopped():
                        terminal.update(status="Finishing transcript")
                    case RealtimeTranscriptDelta(delta=delta):
                        partial_transcript += delta
                        terminal.update(
                            status="Transcribing",
                            transcript=partial_transcript,
                        )
                    case RealtimeTranscriptCompleted(transcript=transcript):
                        terminal.commit("transcript", transcript)
                        terminal.update(status="Completed")
                        return
                    case RealtimeError(error=error):
                        terminal.error(str(error))
                        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
