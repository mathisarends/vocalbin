"""Runnable Cartesia text-to-speech examples.

Run with valid CARTESIA_API_KEY and CARTESIA_VOICE_ID environment variables:

    uv run --extra cartesia python examples/cartesia/text_to_speech.py

Generated audio is written to examples/output/. Streaming examples use raw
16-bit PCM at 24 kHz.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv

from vocalbin.cartesia import (
    CartesiaGenerationConfig,
    CartesiaTextToSpeech,
    CartesiaTextToSpeechConfig,
    CartesiaTextToSpeechRequest,
    CartesiaWavOutputFormat,
)

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "output"


async def synthesize(voice_id: str) -> None:
    async with CartesiaTextToSpeech() as tts:
        response = await tts.synthesize(
            CartesiaTextToSpeechRequest(
                text="Hallo aus vocalbin mit Cartesia!",
                voice_id=voice_id,
                language="de",
                output_format=CartesiaWavOutputFormat(),
            )
        )

    _save(response.audio, "cartesia_synthesize.wav")
    print(f"synthesize: {len(response.audio)} bytes, {response.content_type}")


async def stream(voice_id: str) -> None:
    request = CartesiaTextToSpeechRequest(
        text="Dieser vollständige Text wird als Audiostream übertragen.",
        voice_id=voice_id,
        language="de",
    )

    audio = bytearray()
    async with CartesiaTextToSpeech() as tts:
        async for chunk in tts.stream(request):
            audio.extend(chunk)

    _save(audio, "cartesia_stream.pcm")
    print(f"stream: {len(audio)} bytes")


async def stream_text(voice_id: str) -> None:
    async def text_chunks() -> AsyncIterator[str]:
        for chunk in (
            "Dieser Text wird ",
            "Stück für Stück ",
            "an Cartesia gesendet.",
        ):
            yield chunk

    config = CartesiaTextToSpeechConfig(
        voice_id=voice_id,
        language="de",
        generation_config=CartesiaGenerationConfig(speed=0.9),
    )

    audio = bytearray()
    async with CartesiaTextToSpeech() as tts:
        async for chunk in tts.stream_text(text_chunks(), config):
            audio.extend(chunk)

    _save(audio, "cartesia_stream_text.pcm")
    print(f"stream_text: {len(audio)} bytes")


def _save(audio: bytes | bytearray, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_bytes(audio)


async def main() -> None:
    voice_id = os.environ["CARTESIA_VOICE_ID"]
    await synthesize(voice_id)
    await stream(voice_id)
    await stream_text(voice_id)


if __name__ == "__main__":
    asyncio.run(main())
