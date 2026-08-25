"""Flux multilingual transcription with language hints.

Aura 2 speaks a German and an English line; `flux-general-multi` transcribes both
over a single connection and reports the language it heard per turn.

Run with a valid DEEPGRAM_API_KEY environment variable:

    uv run --extra deepgram python examples/deepgram/flux_multilingual.py
"""

import asyncio
from collections.abc import AsyncIterator

from dotenv import load_dotenv

from vocalbin.deepgram import (
    AuraVoice,
    FluxModel,
    StreamingSpeechToText,
    TextToSpeech,
    events,
)

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SIZE = 3200
LINES = [
    (AuraVoice.AURA_2_JULIUS_DE, "Guten Morgen, hier spricht vocalbin."),
    (AuraVoice.AURA_2_THALIA_EN, "And this sentence is spoken in English."),
]


async def paced_audio(audio: bytes) -> AsyncIterator[bytes]:
    for offset in range(0, len(audio), CHUNK_SIZE):
        yield audio[offset : offset + CHUNK_SIZE]
        await asyncio.sleep(0.1)


async def speak() -> bytes:
    audio = bytearray()
    async with TextToSpeech(sample_rate=SAMPLE_RATE) as text_to_speech:
        for voice, line in LINES:
            response = await text_to_speech.generate(line, model=voice)
            audio.extend(response.audio)
    return bytes(audio)


async def main() -> None:
    audio = await speak()

    async with StreamingSpeechToText(
        model=FluxModel.FLUX_GENERAL_MULTI,
        language_hints=["de", "en"],
        sample_rate=SAMPLE_RATE,
        eot_threshold=0.8,
    ) as speech_to_text:
        async for event in speech_to_text.stream(paced_audio(audio)):
            match event:
                case events.TurnEnd(transcript=transcript, languages=languages):
                    print(f"{languages}: {transcript!r}")


if __name__ == "__main__":
    asyncio.run(main())
