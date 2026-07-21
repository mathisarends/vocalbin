# Contributing to vocalbin

Thanks for your interest in improving `vocalbin`. This project is intentionally
small: a typed, asynchronous wrapper around OpenAI's speech-to-text and
text-to-speech endpoints. A small API surface, clear types, few dependencies and
predictable behavior matter more than a long feature list.

## Scope: read the Vision first

Before opening an issue or pull request, please read [`vision.md`](vision.md). It
defines what this library does, what it deliberately does **not** do, and the
guardrails for future changes.

A change is in scope only if you can answer **yes** to the five questions listed
under _"Leitlinie für zukünftige Pull Requests"_ in [`vision.md`](vision.md) — in
short:

1. Does it directly support speech-to-text or text-to-speech via the OpenAI Audio API?
2. Does it fit the existing request/response/port/client boundaries without adding
   application-specific dependencies?
3. Does the public API stay small, async, typed and understandable?
4. Is the added dependency or complexity justified by a concrete benefit?
5. Are behavior, validation and backward compatibility covered by tests?

Work that needs audio processing, framework integration, persistence or
orchestration belongs in the calling application or a separate package, not here.

## Development setup

```bash
uv sync
uv run pytest
```

Tests run offline against fake clients, so no `OPENAI_API_KEY` is required for the
suite. A key is only needed to run the scripts in [`examples/`](examples/) against
the real API.

## Code conventions

These follow [`AGENTS.md`](AGENTS.md):

- Use relative imports for package re-exports in `__init__.py`; use absolute
  imports everywhere else.
- Model provider interfaces with `ABC` and `@abstractmethod`, not `Protocol`.
  Keep the provider ports in `vocalbin/ports.py`.
- Avoid comments and docstrings that merely restate the code. Add a comment only
  when it explains a non-obvious reason, constraint or tradeoff.
- Validate model capabilities up front (see `vocalbin/models.py`) and keep raw
  provider data available on responses rather than discarding it.

## Pull requests

- Keep changes focused; split unrelated work into separate pull requests.
- Add or update tests for any behavior, validation rule or capability change.
- Update the [`README.md`](README.md) and the relevant script(s) in
  [`examples/`](examples/) when you add or change a model, voice, format or
  parameter.
- Make sure `uv run pytest` passes before submitting.

A pull request that widens the project's scope only because the implementation is
technically possible here should be declined or split, as described in
[`vision.md`](vision.md).
