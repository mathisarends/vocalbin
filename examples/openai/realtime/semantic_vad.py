import asyncio

from dotenv import load_dotenv

from examples.openai.realtime._terminal import RealtimeTerminal
from vocalbin.openai.realtime import TranscriberBuilder, events

load_dotenv()


async def main() -> None:
    transcriber = (
        TranscriberBuilder()
        .model("gpt-4o-transcribe")
        .language("de")
        .semantic_vad(eagerness="medium")
        .build()
    )
    partial_transcript = ""

    with RealtimeTerminal(
        title="Live transcription · Semantic VAD",
        fields=["transcript"],
    ) as terminal:
        async with transcriber:
            async for event in transcriber.stream():
                match event:
                    case events.SessionConnected():
                        terminal.update(status="Ready")
                    case events.SpeechStarted():
                        terminal.update(status="Listening")
                    case events.SpeechStopped():
                        terminal.update(status="Semantic turn detected")
                    case events.TranscriptDelta(delta=delta):
                        partial_transcript += delta
                        terminal.update(
                            status="Transcribing",
                            transcript=partial_transcript,
                        )
                    case events.TranscriptCompleted(transcript=transcript):
                        terminal.commit("transcript", transcript)
                        partial_transcript = ""
                        terminal.update(status="Ready")
                    case events.Error(error=error):
                        terminal.error(str(error))
                        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
