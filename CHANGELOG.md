# Changelog

## 0.8.0 - 2026-10-29

### Added

- OpenAI text-to-speech streaming yields audio as it arrives.
- Shared `IncrementalTextToSpeech`, `AsyncClosable`, and `Connectable` ports expose
  incremental text input, async cleanup, and connection preparation.
- Cartesia and Deepgram TTS support `connect()` to prepare reusable WebSocket
  connections before the first request.
- Deepgram incremental TTS supports `flush_on="first_chunk"` and `flush_on="chunk"`
  for earlier phrase output, with flush acknowledgements drained before reuse and
  a per-service flush budget.
- `RESET` restores an individual provider option to its built-in default while
  preserving other constructor defaults.
- `MicrophoneInput(chunk_duration_ms=...)` configures audio batches by duration and
  exposes the effective sample rate, chunk size, and duration.
- The package ships `py.typed` metadata and checks the public API with mypy.
- A [latency guide](docs/latency.md), a cold/warm first-audio benchmark, and
  [initial live measurements](docs/benchmarks/2026-10-29-ttfa.md) document streaming
  latency and connection reuse.

### Fixed

- Per-call options retain unrelated provider defaults; Cartesia calls accept
  configured voices in typed code.
- Realtime transcription drains every committed turn and releases resources when
  startup fails.
- Deepgram validates codec-specific settings, accepts explicit sample rates for
  raw batch audio, and reports the effective REST audio metadata.
- Deepgram file uploads use bounded reads off the event loop and close files on
  completion or cancellation.
- Piper streaming uses bounded buffers with backpressure and avoids blocking the
  event loop during teardown. Explicit config paths take precedence when the
  model path comes from the environment.
- The release workflow uses locked release dependencies and smoke-tests the
  current realtime exports from the built wheel.

[Changes since v0.7.0](https://github.com/mathisarends/vocalbin/compare/v0.7.0...main).
