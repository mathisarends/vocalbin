"""Runnable Piper text-to-speech examples.

Run with a valid PIPER_MODEL_PATH (and optional PIPER_CONFIG_PATH) environment
variable pointing at a downloaded Piper voice:

    uv run --extra piper python examples/piper/text_to_speech.py

Generated audio is written to examples/output/ as raw 16-bit PCM. Pass a
custom text as the first argument:

    uv run --extra piper python examples/piper/text_to_speech.py "Ein anderer Text"
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from vocalbin.piper import PiperTextToSpeech, PiperTextToSpeechRequest

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "output"


async def generate(text: str) -> None:
    async with PiperTextToSpeech() as tts:
        response = await tts.generate(PiperTextToSpeechRequest(text=text))

    path = _save(response.audio, "piper_generate.pcm")
    print(f"generate: {len(response.audio)} bytes at {response.sample_rate} Hz")
    print(f"-> {path}")


async def stream() -> None:
    request = PiperTextToSpeechRequest(
        text="Dieser vollständige Text wird als Audiostream übertragen."
    )

    audio = bytearray()
    async with PiperTextToSpeech() as tts:
        async for chunk in tts.stream(request):
            audio.extend(chunk)

    path = _save(audio, "piper_stream.pcm")
    print(f"stream: {len(audio)} bytes -> {path}")


def _save(audio: bytes | bytearray, name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_bytes(audio)
    return path.resolve()


async def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "Hallo aus vocalbin mit Piper!"
    await generate(text)
    await stream()


if __name__ == "__main__":
    asyncio.run(main())
