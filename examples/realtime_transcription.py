import asyncio

from examples._realtime_terminal import RealtimeTerminal
from vocalbin.realtime import (
    OpenAIRealtimeTranscriber,
    RealtimeError,
    RealtimeSessionConnected,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
)


async def main() -> None:
    config = RealtimeTranscriptionConfig(language="de")
    partial_transcript = ""

    with RealtimeTerminal(
        title="Live transcription · German",
        fields=["transcript"],
    ) as terminal:
        async with OpenAIRealtimeTranscriber(config) as transcriber:
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
                        partial_transcript = ""
                        terminal.update(status="Ready")
                    case RealtimeError(error=error):
                        terminal.error(str(error))
                        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
