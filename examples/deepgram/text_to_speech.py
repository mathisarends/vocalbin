"""Runnable Deepgram text-to-speech examples.

Run with a valid DEEPGRAM_API_KEY environment variable:

    uv run --extra deepgram python examples/deepgram/text_to_speech.py

Generated audio is written to examples/output/. Streaming examples use raw
16-bit PCM at 24 kHz.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv

from vocalbin.deepgram import AudioContainer, TextToSpeech, TextToSpeechModel

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "output"


async def generate() -> None:
    async with TextToSpeech() as tts:
        response = await tts.generate(
            "Hallo aus vocalbin mit Deepgram Aura 2!",
            model=TextToSpeechModel.AURA_2_THALIA_EN,
            container=AudioContainer.WAV,
        )

    _save(response.audio, "deepgram_generate.wav")
    print(f"generate: {len(response.audio)} bytes, {response.content_type}")


async def stream() -> None:
    audio = bytearray()
    async with TextToSpeech() as tts:
        async for chunk in tts.stream(
            "Dieser vollständige Text wird als Audiostream übertragen."
        ):
            audio.extend(chunk)

    _save(audio, "deepgram_stream.pcm")
    print(f"stream: {len(audio)} bytes")


async def stream_incremental() -> None:
    async def text_chunks() -> AsyncIterator[str]:
        for chunk in (
            "Dieser Text wird ",
            "Stück für Stück ",
            "an Deepgram gesendet.",
        ):
            yield chunk

    audio = bytearray()
    async with TextToSpeech() as tts:
        async for chunk in tts.stream_incremental(
            text_chunks(),
            model=TextToSpeechModel.AURA_2_ORION_EN,
        ):
            audio.extend(chunk)

    _save(audio, "deepgram_stream_incremental.pcm")
    print(f"stream_incremental: {len(audio)} bytes")


def _save(audio: bytes | bytearray, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_bytes(audio)


async def main() -> None:
    await generate()
    await stream()
    await stream_incremental()


if __name__ == "__main__":
    asyncio.run(main())
