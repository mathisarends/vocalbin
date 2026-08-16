"""Share one AsyncOpenAI client across both services.

    uv run python examples/openai/shared_client.py

When you pass `client=`, vocalbin does not own or close it. This lets you reuse
a single configured client (custom base_url, timeouts, retries) for both
speech-to-text and text-to-speech, and close it yourself when you are done.
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from vocalbin.openai import (
    SpeechToText,
    SpeechToTextRequest,
    TextToSpeech,
    TextToSpeechFormat,
)

load_dotenv()


async def main() -> None:
    client = AsyncOpenAI()

    tts = TextToSpeech(client=client)
    stt = SpeechToText(client=client)

    spoken = await tts.generate(
        "Ein Client, zwei Dienste.",
        response_format=TextToSpeechFormat.WAV,
    )
    heard = await stt.transcribe(
        SpeechToTextRequest(audio=spoken.audio, filename="shared.wav", language="de")
    )
    print(f"transcribed: {heard.text!r}")

    # The injected client is caller-owned; close it explicitly.
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
