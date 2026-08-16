import asyncio
import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv

from vocalbin.cartesia import RawOutputFormat, SpeechToText, TextToSpeech, events

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SIZE = 3200


async def paced_audio(audio: bytes) -> AsyncIterator[bytes]:
    for offset in range(0, len(audio), CHUNK_SIZE):
        yield audio[offset : offset + CHUNK_SIZE]
        await asyncio.sleep(0.1)


async def main() -> None:
    voice_id = os.environ["CARTESIA_VOICE_ID"]
    async with TextToSpeech() as text_to_speech:
        response = await text_to_speech.generate(
            "Hello from Sonic 3.5. Ink 2 will transcribe this audio in real time.",
            voice_id=voice_id,
            language="en",
            output_format=RawOutputFormat(sample_rate=SAMPLE_RATE),
        )

    async with SpeechToText() as speech_to_text:
        async for event in speech_to_text.stream(paced_audio(response.audio)):
            match event:
                case events.TurnUpdate(transcript=transcript):
                    print(f"update: {transcript}")
                case events.TurnEnd(transcript=transcript):
                    print(f"final: {transcript}")


if __name__ == "__main__":
    asyncio.run(main())
