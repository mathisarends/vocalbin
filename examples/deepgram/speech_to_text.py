"""Runnable Deepgram speech-to-text examples.

Run with a valid DEEPGRAM_API_KEY environment variable:

    uv run --extra deepgram python examples/deepgram/speech_to_text.py

The examples need an audio file. If examples/output/deepgram_sample.wav does not
exist, it is generated once via text-to-speech so every example is
self-contained.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from vocalbin.deepgram import (
    AudioContainer,
    SpeechToText,
    SpeechToTextModel,
    TextToSpeech,
)

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "output"
SAMPLE = OUTPUT_DIR / "deepgram_sample.wav"
SAMPLE_TEXT = "Good morning, this is a transcription test for vocalbin."


async def from_path() -> None:
    async with SpeechToText() as stt:
        response = await stt.transcribe(SAMPLE, smart_format=True)
    print(f"from_path: {response.text!r}, confidence {response.confidence}")


async def from_bytes() -> None:
    async with SpeechToText() as stt:
        response = await stt.transcribe(SAMPLE.read_bytes())
    print(f"from_bytes: {response.text!r}")


async def with_keyterms() -> None:
    async with SpeechToText(model=SpeechToTextModel.NOVA_3) as stt:
        response = await stt.transcribe(
            SAMPLE,
            keyterms=["vocalbin"],
            punctuate=True,
            diarize=True,
        )
    print(f"with_keyterms: {response.text!r}, request {response.request_id}")


async def ensure_sample() -> None:
    if SAMPLE.exists():
        return
    async with TextToSpeech() as tts:
        response = await tts.generate(SAMPLE_TEXT, container=AudioContainer.WAV)
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAMPLE.write_bytes(response.audio)


async def main() -> None:
    await ensure_sample()
    await from_path()
    await from_bytes()
    await with_keyterms()


if __name__ == "__main__":
    asyncio.run(main())
