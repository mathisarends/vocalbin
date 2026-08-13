import asyncio

from examples.openai.realtime._terminal import RealtimeTerminal
from vocalbin.openai.realtime import (
    OpenAIRealtimeTranscriber,
    RealtimeError,
    RealtimeSessionConnected,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionModel,
    SemanticVadConfig,
)


async def main() -> None:
    config = RealtimeTranscriptionConfig(
        model=RealtimeTranscriptionModel.GPT_4O_TRANSCRIBE,
        language="de",
        turn_detection=SemanticVadConfig(eagerness="medium"),
    )
    partial_transcript = ""

    with RealtimeTerminal(
        title="Live transcription · Semantic VAD",
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
                        terminal.update(status="Semantic turn detected")
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
