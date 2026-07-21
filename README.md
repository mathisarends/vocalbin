# vocalbin

`vocalbin` is a small, typed, asynchronous wrapper around OpenAI's speech-to-text
and text-to-speech endpoints. It has no dependency on application-specific settings
or domain packages.

## Installation

```bash
uv add vocalbin
```

Set `OPENAI_API_KEY`, or pass an API key directly when creating a service.

## Speech to text

```python
from pathlib import Path

from vocalbin import OpenAISpeechToText, SpeechToTextRequest


async def transcribe() -> str:
    async with OpenAISpeechToText() as speech_to_text:
        response = await speech_to_text.transcribe(
            SpeechToTextRequest(audio_path=Path("speech.wav"), language="de")
        )
    return response.text
```

Audio can also be supplied directly as bytes:

```python
request = SpeechToTextRequest(audio=audio_bytes, filename="speech.wav")
```

## Text to speech

```python
from vocalbin import (
    OpenAITextToSpeech,
    TextToSpeechFormat,
    TextToSpeechRequest,
    TextToSpeechVoice,
)


async def synthesize() -> bytes:
    async with OpenAITextToSpeech() as text_to_speech:
        response = await text_to_speech.synthesize(
            TextToSpeechRequest(
                text="Hallo aus vocalbin!",
                voice=TextToSpeechVoice.MARIN,
                response_format=TextToSpeechFormat.MP3,
                instructions="Sprich ruhig und freundlich.",
            )
        )
    return response.audio
```

The provider-independent `SpeechToText` and `TextToSpeech` interfaces are abstract
base classes. Both concrete services also accept an existing `AsyncOpenAI` instance
through `client=`. Injected clients remain owned by the caller and are not closed by
`vocalbin`.
