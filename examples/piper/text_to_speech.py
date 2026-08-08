"""Runnable Piper text-to-speech examples.

Run with a valid PIPER_MODEL_PATH (and optional PIPER_CONFIG_PATH) environment
variable pointing at a downloaded Piper voice:

    uv run --extra piper python examples/piper/text_to_speech.py

Generated audio is written to examples/output/ as raw 16-bit PCM.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from vocalbin.piper import PiperTextToSpeech, PiperTextToSpeechRequest

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "output"


async def synthesize() -> None:
    async with PiperTextToSpeech() as tts:
        response = await tts.synthesize(
            PiperTextToSpeechRequest(text="Hallo aus vocalbin mit Piper!")
        )

    _save(response.audio, "piper_synthesize.pcm")
    print(f"synthesize: {len(response.audio)} bytes at {response.sample_rate} Hz")


async def stream() -> None:
    request = PiperTextToSpeechRequest(
        text="Dieser vollständige Text wird als Audiostream übertragen."
    )

    audio = bytearray()
    async with PiperTextToSpeech() as tts:
        async for chunk in tts.stream(request):
            audio.extend(chunk)

    _save(audio, "piper_stream.pcm")
    print(f"stream: {len(audio)} bytes")


def _save(audio: bytes | bytearray, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_bytes(audio)


async def main() -> None:
    await synthesize()
    await stream()


if __name__ == "__main__":
    asyncio.run(main())
