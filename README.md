# 🎙️ vocalbin

![vocalbin — typed, async voice for OpenAI](static/banner.png)

`vocalbin` is a small, typed, asynchronous wrapper around OpenAI's speech,
realtime transcription, and realtime translation APIs. It validates model
capabilities up front, normalizes responses without discarding raw data, and stays
independent of any application-specific settings or domain code.

## Inhaltsverzeichnis

- [Installation](#installation)
- [Speech to text](#speech-to-text)
- [Text to speech](#text-to-speech)
- [Realtime transcription](#realtime-transcription)
- [Realtime translation](#realtime-translation)
- [Supported models, voices and formats](#supported-models-voices-and-formats)
- [Examples](#examples)
- [Bring your own client](#bring-your-own-client)
- [Ports](#ports)
- [Development](#development)

## Installation

```bash
uv add vocalbin
```

Realtime support is optional so the base package does not install a WebSocket
stack:

```bash
uv add "vocalbin[realtime]"  # custom audio input
uv add "vocalbin[audio]"     # WebSockets plus microphone input
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

## Realtime transcription

Realtime transcription uses `gpt-realtime-whisper` and streams partial and final
transcripts. Its public API is grouped under `vocalbin.realtime`:

```python
from vocalbin.realtime import (
    OpenAIRealtimeTranscriber,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
)


async def transcribe_live() -> None:
    async with OpenAIRealtimeTranscriber(
        RealtimeTranscriptionConfig(language="de")
    ) as transcriber:
        async for event in transcriber.stream():
            match event:
                case RealtimeTranscriptDelta(delta=delta):
                    print(delta, end="", flush=True)
                case RealtimeTranscriptCompleted(transcript=transcript):
                    print(f"\n{transcript}")
```

The default `MicrophoneInput` sends raw 24 kHz mono PCM16 chunks. Pass an
`AudioInput` implementation or wrap an async byte source with `AudioStreamInput`
from `vocalbin.realtime` when audio already comes from a media pipeline.
`flush()` manually commits the current transcription buffer.

## Realtime translation

Live interpretation uses the dedicated `gpt-realtime-translate` endpoint. It
continuously returns translated 24 kHz PCM16 audio and target-language transcript
deltas. Optional source-language transcripts use `gpt-realtime-whisper` on the
same session:

```python
from vocalbin.realtime import (
    OpenAIRealtimeTranslator,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationConfig,
    RealtimeTranslationLanguage,
    RealtimeTranslationTranscriptDelta,
)


async def translate_live() -> None:
    config = RealtimeTranslationConfig(
        target_language=RealtimeTranslationLanguage.ENGLISH
    )
    translated_audio = bytearray()

    async with OpenAIRealtimeTranslator(config) as translator:
        async for event in translator.stream():
            match event:
                case RealtimeTranslationTranscriptDelta(delta=delta):
                    print(delta, end="", flush=True)
                case RealtimeTranslationAudioDelta(audio=audio):
                    translated_audio.extend(audio)
```

Translation sessions have no assistant turns and do not use `response.create`.
For finite custom inputs, vocalbin sends `session.close` after the last chunk and
keeps draining output until `session.closed`.

The same realtime namespace also provides audio inputs, providers, shared events,
and session enums:

```python
from vocalbin.realtime import (
    AudioInput,
    AudioStreamInput,
    MicrophoneInput,
    OpenAIRealtimeProvider,
    RealtimeError,
    RealtimeNoiseReduction,
    RealtimeSessionConnected,
    RealtimeSessionType,
)
```

## Supported models, voices and formats

**Speech to text** — `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
`gpt-4o-transcribe-diarize`, `whisper-1`. Response formats and options are
validated per model (for example, `timestamp_granularities` require `whisper-1`
with `verbose_json`, and `include=["logprobs"]` requires a GPT transcription model
with `json`).

**Text to speech** — `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`; output formats `mp3`,
`opus`, `aac`, `flac`, `wav`, `pcm`. The legacy `tts-1`/`tts-1-hd` models accept
only the legacy voices and do not support `instructions`.

**Realtime** — `gpt-realtime-whisper` for live transcription and
`gpt-realtime-translate` for live speech-to-speech translation. Translation
targets are English, Spanish, Portuguese, French, Japanese, Russian, Chinese,
German, Korean, Hindi, Indonesian, Vietnamese, and Italian.

## Examples

The [`examples/`](examples/) directory holds runnable, integration-testable scripts
that exercise every model/voice/format combination and double as documentation.
With a valid `OPENAI_API_KEY` set:

```bash
uv run python examples/text_to_speech.py   # every TTS model, voice and format
uv run python examples/speech_to_text.py   # every STT model and response format
uv run python examples/round_trip.py       # synthesize -> transcribe, self-checking
uv run python examples/shared_client.py    # one AsyncOpenAI client for both services
uv run python examples/realtime_transcription.py
uv run python examples/realtime_translation.py
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
classes (`vocalbin/ports.py`); the realtime ports `AudioInput`, `RealtimeProvider`,
`RealtimeTranscription` and `RealtimeTranslation` live in `vocalbin/realtime/ports.py`.
They mark the boundary of the library, so callers can depend on the interface
rather than the OpenAI implementation.

## Development

```bash
uv sync
uv run pytest
```
