# First-audio latency

Use `stream()` when text is already available and `stream_incremental()` when a
supported provider should receive text as it arrives. Forward audio chunks as soon
as they arrive instead of collecting them in a `bytes` object.

## Reuse a prepared connection

Cartesia and Deepgram TTS implement `vocalbin.Connectable`. Create one service per
application/session, call `connect()` before the first turn, and close it during
shutdown. Connecting opens the socket without generating speech.

```python
from contextlib import aclosing

from vocalbin.deepgram import TextToSpeech

async with TextToSpeech(flush_on="first_chunk") as tts:
    await tts.connect()
    async with aclosing(tts.stream_incremental(phrase_stream)) as audio:
        async for chunk in audio:
            await send_to_player(chunk)
```

Deepgram reuses the socket after `connect()`, serializes streams on that socket,
and reconnects if model, encoding or sample rate changes. Cancellation, an error,
or closing a stream early discards its socket. The next request reconnects;
requests that fail after sending text are not automatically replayed. Without
`connect()`, the existing connection-per-stream behavior is preserved.

Cartesia supports multiple contexts on one connection. `connect()` prepares the
same socket later streams use. Its `max_buffer_delay_ms` option controls the text
buffer. Start comparisons at `0` for complete phrases or try a small buffer for
incoming tokens, and compare naturalness as well as latency.

For OpenAI, reuse the same `TextToSpeech`/injected SDK client and choose
`response_format=TextToSpeechFormat.PCM` when the consumer accepts raw PCM. The
wrapper passes HTTP response chunks through without waiting for the complete
response. For Piper, load and retain the voice/service before handling requests;
model loading is separate from synthesis latency.

## Choose when Deepgram flushes

`flush_on="end"` retains the existing behavior. `"first_chunk"` flushes the first
non-empty input chunk immediately, then flushes the rest at EOF. `"chunk"` flushes
every non-empty input chunk. A single-chunk request sends only one flush in every
mode. Set the policy on the constructor or override it on either stream method.
The policy is a local control and can be combined with `config=`.

Feed complete phrases to the early-flush modes. Sending one token at a time can
hurt prosody and exhaust [Deepgram's 20-flush/60-second budget](https://developers.deepgram.com/docs/tts-ws-flush).
The limiter is shared by all streams on one service, including reconnects, and
waits before sending a flush when that budget is exhausted. Every acknowledgement
is drained before the connection is reused.

## Run the benchmark

The repository tool reports JSON and makes real, billable requests for the
selected remote provider. It does not play or save audio. Use the same text,
voice/model, encoding, machine and network for comparisons.

```console
uv run --extra deepgram python scripts/benchmark_tts.py --provider deepgram --runs 10
uv run --extra cartesia python scripts/benchmark_tts.py --provider cartesia --voice YOUR_VOICE_ID --buffer-delay-ms 0 --runs 10
uv run python scripts/benchmark_tts.py --provider openai --runs 10
uv run --extra piper python scripts/benchmark_tts.py --provider piper --model-path voices/en.onnx --runs 10
```

`--mode cold` creates a new service for each sample. `--mode warm` retains one
service, preconnects where supported, and drains `--warmup` requests before taking
samples. `--mode both` is the default. These labels describe client/connection
lifecycle; they cannot force a remote model into a cold or warm server state.

Deepgram runs are automatically spaced outside the measured requests, using the
expected flush count, to avoid exhausting its rolling flush budget during normal
comparisons. The report records `request_pause_ms`. Set `--pause-ms 0` to measure
bursts including limiter waits, or choose another gap explicitly. Other providers
have no automatic pause. Request timeouts must cover any simulated delays and
intentional rate-limit waits.

For incremental text, pass explicit phrase chunks and optional simulated arrival
delays. The text producer runs concurrently with connection setup.

```console
uv run --extra deepgram python scripts/benchmark_tts.py --provider deepgram --text-chunks "Hello! " "How can I help?" --chunk-delay-ms 300 --flush-on first_chunk --runs 10
uv run --extra deepgram python scripts/benchmark_tts.py --provider deepgram --text-chunks "Hello! " "How can I help?" --chunk-delay-ms 300 --flush-on end --runs 10
```

Measurements:

- `first_audio_ms`: start of stream consumption to the first non-empty audio chunk,
  including connection setup and simulated text delays when applicable.
- `first_text_ms`: availability of the first input chunk relative to that start;
  zero when complete text is supplied.
- `text_to_first_audio_ms`: first audio minus first input availability.
- `total_ms` and `audio_bytes`: full stream consumption, with no playback or file
  writes in the measured loop.
- `setup_ms`: client creation/model loading, reported separately from TTFA.
- `connect_ms`: explicit preparation for warm WebSocket runs, outside TTFA.

The summary reports the median and nearest-rank p95 of `first_audio_ms`. Ten runs
are a quick check; use more samples for a useful p95. Empty responses and failed
or timed-out requests fail the benchmark rather than producing favorable samples.
The configured timeout applies to each request and to explicit connection setup.

Raw PCM prevents a container header from being counted as the first audio data.
Audio byte arrival and audible playback are separate measurements: decoder,
resampling, network forwarding and playback prefill can add latency afterwards.
When the library is used through a server, measure time at the final consumer too
so proxy/server buffering is included.
