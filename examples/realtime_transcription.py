import asyncio

from vocalbin.transcription import (
    OpenAIRealtimeTranscriber,
    RealtimeError,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
)


async def main() -> None:
    config = RealtimeTranscriptionConfig(language="de")

    async with OpenAIRealtimeTranscriber(config) as transcriber:
        async for event in transcriber.stream():
            match event:
                case RealtimeTranscriptDelta(delta=delta):
                    print(delta, end="", flush=True)
                case RealtimeTranscriptCompleted(transcript=transcript):
                    print(f"\n>>> {transcript}\n")
                case RealtimeError(error=error):
                    print(f"\nRealtime error: {error}")


if __name__ == "__main__":
    asyncio.run(main())
