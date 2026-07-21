# vocalbin

`vocalbin` is a small, typed, asynchronous wrapper around OpenAI's speech-to-text
and text-to-speech endpoints. It validates model capabilities up front, normalizes
responses without discarding raw data, and stays independent of any
application-specific settings or domain code.

## Installation

```bash
uv add vocalbin
```

Set `OPENAI_API_KEY` in the environment, or pass an API key directly when creating
a service. The default path reads the environment through `OpenAICredentials`:

```python
from vocalbin import OpenAICredentials

credentials = OpenAICredentials()
api_key = credentials.api_key.get_secret_value()
```

An explicit `api_key` takes precedence over the environment. An injected
`AsyncOpenAI` client does not load credentials at all.

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

Audio can also be supplied directly as bytes; `filename` only sets the multipart
upload name:

```python
request = SpeechToTextRequest(audio=audio_bytes, filename="speech.wav")
```

Every request carries the transcript on `response.text` and the untouched provider
payload on `response.raw` (a `dict` for JSON-like formats, a `str` for `text`,
`srt` and `vtt`).

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

`response.content_type` gives the matching MIME type (e.g. `audio/mpeg`).

## Supported models, voices and formats

**Speech to text** — `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, `whisper-1`. Response formats and options are
validated per model (for example, `timestamp_granularities` require `whisper-1`
with `verbose_json`, and `include=["logprobs"]` requires a GPT transcription model
with `json`).

**Text to speech** — `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`; output formats `mp3`,
`opus`, `aac`, `flac`, `wav`, `pcm`. The legacy `tts-1`/`tts-1-hd` models accept
only the legacy voices and do not support `instructions`.

## Examples

The [`examples/`](examples/) directory holds runnable, integration-testable scripts
that exercise every model/voice/format combination and double as documentation.
With a valid `OPENAI_API_KEY` set:

```bash
uv run python examples/text_to_speech.py   # every TTS model, voice and format
uv run python examples/speech_to_text.py   # every STT model and response format
uv run python examples/round_trip.py       # synthesize -> transcribe, self-checking
uv run python examples/shared_client.py    # one AsyncOpenAI client for both services
```

Generated audio and transcripts are written to `examples/output/` (git-ignored).
`speech_to_text.py` synthesizes its own `sample.wav` on first run, so it needs no
external audio file.

## Bring your own client

Both concrete services accept an existing `AsyncOpenAI` instance via `client=`,
which lets you share one configured client (custom `base_url`, timeouts, retries)
across both services. Injected clients remain owned by the caller and are not
closed by `vocalbin`:

```python
from openai import AsyncOpenAI

from vocalbin import OpenAISpeechToText, OpenAITextToSpeech

client = AsyncOpenAI()
tts = OpenAITextToSpeech(client=client)
stt = OpenAISpeechToText(client=client)
# ... use both, then close it yourself:
await client.close()
```

## Ports

The provider-independent `SpeechToText` and `TextToSpeech` ports are abstract base
classes (`vocalbin/ports.py`). They mark the boundary of the library, so callers
can depend on the interface rather than the OpenAI implementation.

## Development

```bash
uv sync
uv run pytest
```
