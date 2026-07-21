"""Round trip: synthesize text, transcribe it back, and compare.

This doubles as a smoke test for a live OPENAI_API_KEY:

    uv run python examples/round_trip.py

It fails loudly if the transcript does not contain the key phrase, so it is a
cheap end-to-end check that both endpoints are reachable and wired correctly.
"""

import asyncio

from vocalbin import (
    OpenAISpeechToText,
    OpenAITextToSpeech,
    SpeechToTextRequest,
    TextToSpeechFormat,
    TextToSpeechRequest,
)

PHRASE = "vocalbin"
TEXT = f"Dies ist ein Round-Trip-Test von {PHRASE}."


async def main() -> None:
    async with OpenAITextToSpeech() as tts:
        spoken = await tts.synthesize(
            TextToSpeechRequest(text=TEXT, response_format=TextToSpeechFormat.WAV)
        )

    async with OpenAISpeechToText() as stt:
        heard = await stt.transcribe(
            SpeechToTextRequest(
                audio=spoken.audio,
                filename="round_trip.wav",
                language="de",
            )
        )

    print(f"spoken:      {TEXT!r}")
    print(f"transcribed: {heard.text!r}")

    if PHRASE.lower() not in heard.text.lower():
        raise SystemExit(f"round trip failed: {PHRASE!r} not found in transcript")
    print("round trip ok")


if __name__ == "__main__":
    asyncio.run(main())
