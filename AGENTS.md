# Repository conventions

- Use relative imports for package re-exports in `__init__.py`; use absolute imports in all other modules.
- Model provider interfaces with `ABC` and `@abstractmethod`, not `Protocol`.
- Keep provider ports in `vocalbin/ports.py`.
- Keep credential loading in `vocalbin/credentials.py` and use `pydantic-settings`.
- Avoid comments and docstrings that merely restate what the code already says.
- Add a comment only when it explains a non-obvious reason, constraint, or tradeoff.
- Do not add module docstrings that only summarize the module name or contents.
