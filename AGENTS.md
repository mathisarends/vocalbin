# Repository conventions

- Use relative imports for package re-exports in `__init__.py`; use absolute imports in all other modules.
- Model provider interfaces with `ABC` and `@abstractmethod`, not `Protocol`.
- Keep provider ports beside the models they describe: speech ports in
  `vocalbin/ports.py`, realtime ports in `vocalbin/openai/realtime/ports.py`.
- Keep the realtime package self-contained: it must not import from
  `vocalbin/openai/models.py` or `vocalbin/ports.py`.
- Keep credential loading beside each provider's client (e.g.
  `vocalbin/openai/credentials.py`, `vocalbin/cartesia/credentials.py`) and use
  `pydantic-settings`. Providers with no remote credentials (e.g. the local
  `vocalbin/piper/config.py`) follow the same beside-the-client,
  `pydantic-settings` pattern but are named `config.py`, not `credentials.py`.
- Declare `__all__` only in `__init__.py`; regular modules must not repeat their
  public names in a trailing `__all__`.
- Avoid comments and docstrings that merely restate what the code already says.
- Add a comment only when it explains a non-obvious reason, constraint, or tradeoff.
- Do not add module docstrings that only summarize the module name or contents.
