"""Aura 2 speech feeding Flux conversational transcription.

Run with a valid DEEPGRAM_API_KEY environment variable:

    uv run --extra deepgram python examples/deepgram/round_trip.py
"""

import asyncio
from collections.abc import AsyncIterator

from dotenv import load_dotenv

from vocalbin.deepgram import StreamingSpeechToText, TextToSpeech, events

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SIZE = 3200


async def paced_audio(audio: bytes) -> AsyncIterator[bytes]:
    for offset in range(0, len(audio), CHUNK_SIZE):
        yield audio[offset : offset + CHUNK_SIZE]
        await asyncio.sleep(0.1)


async def main() -> None:
    async with TextToSpeech(sample_rate=SAMPLE_RATE) as text_to_speech:
        response = await text_to_speech.generate(
            "Hello from Aura 2. Flux will transcribe this audio in real time."
        )

    async with StreamingSpeechToText(
        sample_rate=SAMPLE_RATE,
        eager_eot_threshold=0.6,
        eot_threshold=0.8,
    ) as speech_to_text:
        async for event in speech_to_text.stream(paced_audio(response.audio)):
            match event:
                case events.TurnUpdate(transcript=transcript) if transcript:
                    print(f"update: {transcript}")
                case events.TurnEagerEnd(transcript=transcript):
                    print(f"eager end: {transcript}")
                case events.TurnEnd(transcript=transcript):
                    print(f"final: {transcript}")


if __name__ == "__main__":
    asyncio.run(main())
