"""Runnable text-to-speech examples covering every model, voice and format.

Run with a valid OPENAI_API_KEY:

    uv run python examples/text_to_speech.py

Generated audio is written to examples/output/.
"""

import asyncio
from pathlib import Path

from vocalbin import (
    OpenAITextToSpeech,
    TextToSpeechFormat,
    TextToSpeechModel,
    TextToSpeechRequest,
    TextToSpeechVoice,
)

OUTPUT_DIR = Path(__file__).parent / "output"

# tts-1 and tts-1-hd only accept these voices and do not accept `instructions`.
LEGACY_VOICES = [
    TextToSpeechVoice.ALLOY,
    TextToSpeechVoice.ASH,
    TextToSpeechVoice.CORAL,
    TextToSpeechVoice.ECHO,
    TextToSpeechVoice.FABLE,
    TextToSpeechVoice.NOVA,
    TextToSpeechVoice.ONYX,
    TextToSpeechVoice.SAGE,
    TextToSpeechVoice.SHIMMER,
]


async def basic() -> None:
    """Defaults: gpt-4o-mini-tts, voice 'marin', mp3."""
    async with OpenAITextToSpeech() as tts:
        response = await tts.synthesize(
            TextToSpeechRequest(text="Hallo aus vocalbin!")
        )
    _save(response.audio, "basic.mp3")
    print(f"basic: {len(response.audio)} bytes, {response.content_type}")


async def with_instructions_and_speed() -> None:
    """gpt-4o-mini-tts supports free-form voice instructions and a speed factor."""
    async with OpenAITextToSpeech() as tts:
        response = await tts.synthesize(
            TextToSpeechRequest(
                text="Diese Stimme klingt ruhig, freundlich und ein wenig langsamer.",
                model=TextToSpeechModel.GPT_4O_MINI_TTS,
                voice=TextToSpeechVoice.CEDAR,
                instructions="Sprich ruhig, warm und mit leichter Betonung.",
                speed=0.9,
            )
        )
    _save(response.audio, "instructions_speed.mp3")
    print(f"instructions+speed: {len(response.audio)} bytes")


async def every_format() -> None:
    """gpt-4o-mini-tts can emit every supported container/codec."""
    async with OpenAITextToSpeech() as tts:
        for fmt in TextToSpeechFormat:
            response = await tts.synthesize(
                TextToSpeechRequest(
                    text=f"Dies ist das Format {fmt.value}.",
                    response_format=fmt,
                )
            )
            _save(response.audio, f"format_{fmt.value}.{fmt.value}")
            print(f"format {fmt.value}: {response.content_type}")


async def every_voice() -> None:
    """gpt-4o-mini-tts accepts every voice in the enum."""
    async with OpenAITextToSpeech() as tts:
        for voice in TextToSpeechVoice:
            response = await tts.synthesize(
                TextToSpeechRequest(
                    text=f"Das ist die Stimme {voice.value}.",
                    voice=voice,
                )
            )
            _save(response.audio, f"voice_{voice.value}.mp3")
            print(f"voice {voice.value}: {len(response.audio)} bytes")


async def legacy_models() -> None:
    """tts-1 and tts-1-hd: legacy voices only, no instructions, speed allowed."""
    async with OpenAITextToSpeech() as tts:
        for model in (TextToSpeechModel.TTS_1, TextToSpeechModel.TTS_1_HD):
            response = await tts.synthesize(
                TextToSpeechRequest(
                    text="Dies ist ein Legacy-Text-to-Speech-Modell.",
                    model=model,
                    voice=LEGACY_VOICES[0],
                    speed=1.1,
                )
            )
            _save(response.audio, f"legacy_{model.value}.mp3")
            print(f"legacy {model.value}: {len(response.audio)} bytes")


def _save(audio: bytes, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_bytes(audio)


async def main() -> None:
    await basic()
    await with_instructions_and_speed()
    await every_format()
    await every_voice()
    await legacy_models()


if __name__ == "__main__":
    asyncio.run(main())
