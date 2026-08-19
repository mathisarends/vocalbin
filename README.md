# 🎙️ vocalbin

![vocalbin — typed, async voice APIs](static/banner.png)

`vocalbin` is a small, typed, asynchronous wrapper around OpenAI, Cartesia,
Deepgram, and Piper speech APIs. It validates known model capabilities up front,
forwards future model IDs as strings, normalizes responses without discarding
useful data, and stays independent of application-specific settings or domain
code.

## Inhaltsverzeichnis

- [Installation](#installation)
- [Speech to text](#speech-to-text)
- [Text to speech](#text-to-speech)
- [Cartesia text to speech](#cartesia-text-to-speech)
- [Cartesia realtime speech to text](#cartesia-realtime-speech-to-text)
- [Deepgram speech to text](#deepgram-speech-to-text)
- [Deepgram text to speech](#deepgram-text-to-speech)
- [Piper text to speech](#piper-text-to-speech)
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
uv add "vocalbin[cartesia]"  # Cartesia TTS and realtime STT
uv add "vocalbin[deepgram]"  # Deepgram Nova 3, Flux and Aura 2
uv add "vocalbin[piper]"     # Piper local/offline TTS
```

Set `OPENAI_API_KEY` in the environment, or pass an API key directly when creating
a service. The default path reads the environment through `Credentials`:

```python
from vocalbin.openai import Credentials

credentials = Credentials()
api_key = credentials.api_key.get_secret_value()
```

An explicit `api_key` takes precedence over the environment. An injected
`AsyncOpenAI` client does not load credentials at all.

## Speech to text

```python
from pathlib import Path

from vocalbin.openai import SpeechToText


async def transcribe() -> str:
    async with SpeechToText() as speech_to_text:
        response = await speech_to_text.transcribe(Path("speech.wav"), language="de")
    return response.text
```

Audio can also be supplied directly as bytes; `filename` only sets the multipart
upload name:

```python
response = await speech_to_text.transcribe(
    audio_bytes,
    filename="speech.wav",
    language="de",
)
```

Every response carries the transcript on `response.text` and the untouched provider
payload on `response.raw` (a `dict` for JSON-like formats, a `str` for `text`,
`srt` and `vtt`). Reusable defaults use the same flat parameters on the service
constructor, for example `SpeechToText(language="de")`. A complete
`SpeechToTextConfig` can still be supplied per call with `config=`.

## Text to speech

```python
from vocalbin.openai import (
    TextToSpeech,
    TextToSpeechFormat,
    TextToSpeechVoice,
)


async def generate() -> bytes:
    async with TextToSpeech() as text_to_speech:
        response = await text_to_speech.generate(
            "Hallo aus vocalbin!",
            voice=TextToSpeechVoice.MARIN,
            response_format=TextToSpeechFormat.MP3,
            instructions="Sprich ruhig und freundlich.",
        )
    return response.audio
```

`response.content_type` gives the matching MIME type (e.g. `audio/mpeg`).

## Cartesia text to speech

Cartesia is an alternative text-to-speech provider, grouped under
`vocalbin.cartesia`. Install it with `uv add "vocalbin[cartesia]"` and set
`CARTESIA_API_KEY` in the environment:

```python
from vocalbin.cartesia import (
    TextToSpeech,
    Voice,
    WavOutputFormat,
)


async def generate(voice_id: str = Voice.SKYLAR_FRIENDLY_GUIDE) -> bytes:
    async with TextToSpeech(
        voice_id=voice_id,
        language="de",
        output_format=WavOutputFormat(),
    ) as text_to_speech:
        response = await text_to_speech.generate("Hallo aus vocalbin mit Cartesia!")
    return response.audio
```

`Voice` maps Cartesia's published voice names to their UUIDs. Raw UUID strings
remain supported. Refresh the checked-in mapping after Cartesia adds or renames
voices:

```console
uv run --extra cartesia python scripts/generate_voices.py
```

`TextToSpeech` also implements `StreamingTextToSpeech`. `stream()` returns
one full request as an audio chunk stream; `stream_incremental()` takes an async
iterable of text chunks and streams matching audio back over the same WebSocket
connection, so text can be sent incrementally as it becomes available:

```python
from collections.abc import AsyncIterator

from vocalbin.cartesia import TextToSpeech


async def stream_incremental(
    voice_id: str, text_chunks: AsyncIterator[str]
) -> bytes:
    audio = bytearray()

    async with TextToSpeech() as text_to_speech:
        async for chunk in text_to_speech.stream_incremental(
            text_chunks,
            voice_id=voice_id,
            language="de",
        ):
            audio.extend(chunk)
    return bytes(audio)
```

WebSocket streaming requires `output_format=RawOutputFormat()` (the
default), which returns raw 16-bit PCM audio.

## Cartesia realtime speech to text

`SpeechToText` implements `StreamingSpeechToText` with Cartesia's Ink 2
model and built-in turn detection. It accepts an async stream of raw, mono audio
chunks and emits typed turn lifecycle events:

```python
from collections.abc import AsyncIterator

from vocalbin.cartesia import SpeechToText, events


async def transcribe(audio: AsyncIterator[bytes]) -> None:
    async with SpeechToText(sample_rate=16_000) as speech_to_text:
        async for event in speech_to_text.stream(audio):
            match event:
                case events.TurnUpdate(transcript=transcript):
                    print(transcript)
                case events.TurnEnd(transcript=transcript):
                    print(f"final: {transcript}")
```

The default input is mono `pcm_s16le` at 16 kHz. Other raw PCM encodings,
sample rates, keyterms, and turn-detection thresholds use flat parameters on
the constructor or `stream()`. A complete `SpeechToTextConfig` remains available
as a per-call `config=` override. Audio should arrive at realtime speed in small
chunks (Cartesia recommends about 100 ms). Ink 2 currently supports English only.
Cartesia does not expose Ink 2 through its batch STT endpoint, so this adapter
intentionally has no `transcribe()` method.

## Deepgram speech to text

Deepgram is grouped under `vocalbin.deepgram`. Install it with
`uv add "vocalbin[deepgram]"` and set `DEEPGRAM_API_KEY` in the environment.

`SpeechToText` transcribes complete recordings with Nova 3 over the REST API and
accepts raw bytes or a file path:

```python
from pathlib import Path

from vocalbin.deepgram import SpeechToText


async def transcribe(audio: Path) -> str:
    async with SpeechToText(smart_format=True) as speech_to_text:
        response = await speech_to_text.transcribe(audio, keyterms=["vocalbin"])
    return response.text
```

`keyterms` are Nova 3 only; passing them with an older model raises before the
request is sent. The response keeps the provider payload in `raw` alongside the
normalized `text`, `confidence`, `detected_language`, and `request_id`.

`StreamingSpeechToText` implements the `StreamingSpeechToText` port with
Deepgram's Flux model and its conversational turn detection. It accepts an async stream of raw,
mono audio and yields typed turn events:

```python
from collections.abc import AsyncIterator

from vocalbin.deepgram import StreamingSpeechToText, events


async def transcribe(audio: AsyncIterator[bytes]) -> str:
    async with StreamingSpeechToText(
        sample_rate=16000,
        eager_eot_threshold=0.6,
        eot_threshold=0.8,
    ) as speech_to_text:
        async for event in speech_to_text.stream(audio):
            match event:
                case events.TurnEnd(transcript=transcript):
                    return transcript
    return ""
```

Flux emits `Connected`, `TurnStart`, `TurnUpdate`, `TurnEagerEnd`, `TurnResume`,
and `TurnEnd` events; each turn event carries the running `transcript`, its
`words` with confidences, and `end_of_turn_confidence`. `eager_eot_threshold`
must not exceed `eot_threshold` (default `0.7`), which is validated up front. A
`FatalError` from the socket is raised as `SpeechToTextError` with the provider
error code.

## Deepgram text to speech

`TextToSpeech` speaks with Aura 2 and implements both the request-response and
the streaming port:

```python
from vocalbin.deepgram import AudioContainer, TextToSpeech, TextToSpeechModel


async def generate() -> bytes:
    async with TextToSpeech(
        model=TextToSpeechModel.AURA_2_THALIA_EN,
        container=AudioContainer.WAV,
    ) as text_to_speech:
        response = await text_to_speech.generate("Hallo aus vocalbin mit Deepgram!")
    return response.audio
```

`stream()` sends one full request and yields audio chunks;
`stream_incremental()` takes an async iterable of text chunks and streams
matching audio back over the same WebSocket connection. WebSocket streaming
carries no container and supports `linear16`, `mulaw`, and `alaw` only, so
compressed encodings are rejected before connecting. Deepgram signals problems on
the socket as warnings, which are raised as `TextToSpeechError`.

Both Deepgram clients follow the same WebSocket lifecycle as the Cartesia
clients: lazy connect, `connect()`, `is_connected`, `disconnect()`, and `aclose()`
for terminal cleanup.

## Piper text to speech

[Piper](https://github.com/OHF-Voice/piper1-gpl) is a local, offline
text-to-speech engine, grouped under `vocalbin.piper`. Install it with
`uv add "vocalbin[piper]"`, download a voice model, and point
`PIPER_MODEL_PATH` (and optionally `PIPER_CONFIG_PATH`) at it:

```python
from vocalbin.piper import TextToSpeech


async def generate() -> bytes:
    async with TextToSpeech() as text_to_speech:
        response = await text_to_speech.generate("Hallo aus vocalbin mit Piper!")
    return response.audio
```

`response.audio` is raw 16-bit PCM at the voice model's sample rate
(`response.sample_rate`). `TextToSpeech` also implements
`StreamingTextToSpeech`; `stream()` yields the same raw PCM audio in chunks as
Piper synthesizes it, off the event loop:

```python
async def stream() -> bytes:
    audio = bytearray()
    async with TextToSpeech() as text_to_speech:
        async for chunk in text_to_speech.stream("Dieser Text wird gestreamt."):
            audio.extend(chunk)
    return bytes(audio)
```

Pass an existing `PiperVoice` via `voice=` to reuse an already-loaded model
across requests instead of loading it from `model_path`/credentials each time.

## Realtime transcription

Realtime transcription uses `gpt-realtime-whisper` and streams partial and final
transcripts. Its public API is grouped under `vocalbin.openai.realtime`:

```python
from vocalbin.openai.realtime import TranscriberBuilder, events


async def transcribe_live() -> None:
    transcriber = (
        TranscriberBuilder()
        .model("gpt-4o-transcribe")
        .language("de")
        .semantic_vad(eagerness="medium")
        .build()
    )
    async with transcriber:
        async for event in transcriber.stream():
            match event:
                case events.TranscriptDelta(delta=delta):
                    print(delta, end="", flush=True)
                case events.TranscriptCompleted(transcript=transcript):
                    print(f"\n{transcript}")
```

`TranscriberBuilder` and `TranslatorBuilder` are
standalone objects. Their `build()` methods return the corresponding realtime
service, and both builders can be initialized from an existing config.

The default `MicrophoneInput` sends raw 24 kHz mono PCM16 chunks. Pass an
`ports.AudioInput` implementation or wrap an async byte source with `AudioStreamInput`
from `vocalbin.openai.realtime` when audio already comes from a media pipeline.
With semantic VAD enabled, OpenAI automatically detects completed turns and
commits their transcription buffers. Leave `turn_detection` as `None` and call
`flush()` to commit a buffer manually. `gpt-realtime-whisper` does not support
turn detection; use `gpt-4o-transcribe` for Semantic VAD.

## Realtime translation

Live interpretation uses the dedicated `gpt-realtime-translate` endpoint. It
continuously returns translated 24 kHz PCM16 audio and target-language transcript
deltas. Optional source-language transcripts use `gpt-realtime-whisper` on the
same session:

```python
from vocalbin.openai.realtime import TranslationLanguage, TranslatorBuilder, events


async def translate_live() -> None:
    translator = TranslatorBuilder().target_language("en").build()
    translated_audio = bytearray()

    async with translator:
        async for event in translator.stream():
            match event:
                case events.TranslationTranscriptDelta(delta=delta):
                    print(delta, end="", flush=True)
                case events.TranslationAudioDelta(audio=audio):
                    translated_audio.extend(audio)
```

Translation sessions have no assistant turns and do not use `response.create`.
For finite custom inputs, vocalbin sends `session.close` after the last chunk and
keeps draining output until `session.closed`.

The same realtime namespace also provides audio inputs, providers, shared events,
and session enums:

```python
from vocalbin.openai.realtime import (
    AudioStreamInput,
    MicrophoneInput,
    Provider,
    NoiseReduction,
    SessionType,
    events,
    ports,
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

**Cartesia text to speech** — `sonic-3.5`, `sonic-3`, dated model snapshots, and
`sonic-latest`; output containers `raw` (16-bit PCM, WAV, µ-law or A-law
encoding), `wav`, and `mp3`. WebSocket streaming via `stream()` or
`stream_incremental()` requires the `raw` container.

**Cartesia speech to text** — `ink-2` over realtime WebSockets with native turn
detection. Input encodings are `pcm_s16le`, `pcm_s32le`, `pcm_f16le`,
`pcm_f32le`, `pcm_mulaw`, and `pcm_alaw`; the model currently supports English.

**Deepgram speech to text** — `nova-3`, `nova-3-general`, `nova-3-medical`, and
`nova-2` over REST; `flux-general-en` and `flux-general-multi` over the realtime
WebSocket with native turn detection. Streaming input encodings are `linear16`,
`mulaw`, and `alaw`.

**Deepgram text to speech** — the Aura 2 voices (`aura-2-thalia-en` and the
English and Spanish voices alongside it); encodings `linear16`, `mulaw`, `alaw`,
`mp3`, `opus`, `flac`, and `aac`, optionally wrapped in a `wav` or `ogg`
container. `bit_rate` applies to the compressed encodings only.

**Piper text to speech** — any locally installed Piper voice model (`.onnx` +
`.onnx.json`); output is always raw 16-bit PCM at the voice's native sample
rate. `speaker_id` selects a speaker for multi-speaker models; `length_scale`,
`noise_scale`, and `noise_w_scale` tune speaking rate and expressiveness.

**Realtime** — `gpt-realtime-whisper` for live transcription and
`gpt-realtime-translate` for live speech-to-speech translation. Translation
targets are English, Spanish, Portuguese, French, Japanese, Russian, Chinese,
German, Korean, Hindi, Indonesian, Vietnamese, and Italian.

## Examples

The [`examples/`](examples/) directory holds runnable, integration-testable scripts
that exercise every model/voice/format combination and double as documentation.
Scripts are grouped by provider. OpenAI's realtime transcription and translation
examples and their shared terminal renderer live under `examples/openai/realtime/`.
With a valid `OPENAI_API_KEY` set:

```bash
uv run python examples/openai/text_to_speech.py   # every TTS model, voice and format
uv run python examples/openai/speech_to_text.py   # every STT model and response format
uv run python examples/openai/round_trip.py       # generate -> transcribe, self-checking
uv run python examples/openai/shared_client.py    # one AsyncOpenAI client for both services
uv run python examples/openai/realtime/transcription.py
uv run python examples/openai/realtime/semantic_vad.py
uv run python examples/openai/realtime/translation.py
```

Cartesia's request-response and WebSocket streaming calls are demonstrated in one
TTS script. The STT script generates English test audio with Sonic 3.5 and streams
it into Ink 2. Set `CARTESIA_API_KEY` and `CARTESIA_VOICE_ID`, then run:

```bash
uv run --extra cartesia python examples/cartesia/text_to_speech.py
uv run --extra cartesia python examples/cartesia/speech_to_text.py
uv run --extra cartesia --extra audio python examples/cartesia/round_trip.py
```

`round_trip.py` records one English turn from the microphone, sends it through
Ink 2, simulates a streaming LLM response, and plays the Sonic 3.5 response as it
arrives. Timestamped logs make the latency of each stage visible.

Deepgram's REST and WebSocket calls are demonstrated the same way. The STT
script synthesizes its own sample with Aura 2, and `round_trip.py` streams that
audio into Flux. Set `DEEPGRAM_API_KEY`, then run:

```bash
uv run --extra deepgram python examples/deepgram/text_to_speech.py
uv run --extra deepgram python examples/deepgram/speech_to_text.py
uv run --extra deepgram python examples/deepgram/round_trip.py
```

Piper's request-response and streaming calls are demonstrated the same way.
Set `PIPER_MODEL_PATH` (and optionally `PIPER_CONFIG_PATH`) to a downloaded
voice model, then run:

```bash
uv run --extra piper python examples/piper/text_to_speech.py
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

from vocalbin.openai import SpeechToText, TextToSpeech

client = AsyncOpenAI()
tts = TextToSpeech(client=client)
stt = SpeechToText(client=client)
# ... use both, then close it yourself:
await client.close()
```

## Ports

The provider-independent `SpeechToText` and `TextToSpeech` ports are abstract base
classes (`vocalbin/ports.py`); the realtime ports `ports.AudioInput`, `ports.Provider`,
`ports.Transcription` and `ports.Translation` live in `vocalbin/openai/realtime/ports.py`.
They mark the boundary of the library, so callers can depend on the interface
rather than the OpenAI implementation.

## Development

```bash
uv sync
uv run pytest
```
