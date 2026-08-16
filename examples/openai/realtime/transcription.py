import asyncio

from dotenv import load_dotenv

from examples.openai.realtime._terminal import RealtimeTerminal
from vocalbin.openai.realtime import TranscriberBuilder, events

load_dotenv()


async def main() -> None:
    partial_transcript = ""

    with RealtimeTerminal(
        title="Live transcription · German",
        fields=["transcript"],
    ) as terminal:
        transcriber = (
            TranscriberBuilder()
            .model("gpt-4o-transcribe")
            .language("de")
            .semantic_vad(eagerness="low")
            .build()
        )
        async with transcriber:
            async for event in transcriber.stream():
                match event:
                    case events.SessionConnected():
                        terminal.update(status="Ready")
                    case events.SpeechStarted():
                        terminal.update(status="Listening")
                    case events.SpeechStopped():
                        terminal.update(status="Finishing transcript")
                    case events.TranscriptDelta(delta=delta):
                        partial_transcript += delta
                        terminal.update(
                            status="Transcribing",
                            transcript=partial_transcript,
                        )
                    case events.TranscriptCompleted(transcript=transcript):
                        terminal.commit("transcript", transcript)
                        terminal.update(status="Completed")
                        return
                    case events.Error(error=error):
                        terminal.error(str(error))
                        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
