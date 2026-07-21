# Vision

## What it does

`vocalbin` is a small, standalone Python library for the file-based OpenAI Audio
API. It offers a clean, typed, async API for two things:

- Transcribe audio files or bytes with OpenAI speech-to-text
- Turn text into audio bytes with OpenAI text-to-speech

It wraps OpenAI's SDK behind stable ABC ports, validates model capabilities early,
and normalizes responses without dropping useful raw data. A thin credentials
boundary reads the environment variables the OpenAI client needs. The library stays
independent of any application's settings, web, storage, and domain code.

The goal is to stay small: a minimal API surface, clear types, few dependencies, and
predictable behavior over a long feature list.

## What it does not do

`vocalbin` is not a general audio or voice-agent platform. Out of scope:

- Recording, playback, or device control
- Conversion, cutting, normalization, or other signal processing
- Persistence, uploads, object storage, or databases
- Realtime voice dialogs, orchestration, or telephony
- Web, UI, CLI, or application-specific integrations
- Credentials or settings beyond the env vars the OpenAI client needs
- Application business logic or dependencies on other projects

Provider-independent ports define the boundary. Unifying many speech providers into
one large abstraction is a non-goal.

## Guideline for pull requests

A change fits if it directly improves one of the two core flows while respecting the
library boundary — for example: new OpenAI models, voices, formats, or parameters;
fixes to capability validation and response normalization; better typing, error
handling, or client config; backward-compatible API simplifications; tests, docs,
and packaging.

Before accepting a PR, all of these should be yes:

1. Does it directly support speech-to-text or text-to-speech via the OpenAI Audio API?
2. Does it fit the existing request, response, port, and client boundaries without
   application-specific dependencies?
3. Does the public API stay small, async, typed, and understandable?
4. Is the added dependency or complexity worth it?
5. Are behavior, validation, and backward compatibility covered by tests?

Changes needing audio processing, framework integration, persistence, or
orchestration belong in the calling application or a separate package. Reject or
split a PR that widens scope just because it's technically possible here. What
matters is not whether a feature involves audio, but whether it makes this library
better at its one job.
