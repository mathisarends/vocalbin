from __future__ import annotations

import os
import shutil
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TextIO

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"


@dataclass
class _Field:
    label: str
    value: str = ""


class RealtimeTerminal:
    def __init__(
        self,
        title: str,
        fields: Iterable[str],
        *,
        stream: TextIO = sys.stdout,
    ) -> None:
        self._title = title
        self._fields = {label: _Field(label) for label in fields}
        self._stream = stream
        self._interactive = stream.isatty()
        self._color = self._interactive and "NO_COLOR" not in os.environ
        self._status = "Connecting…"
        self._rendered_lines = 0
        self._closed = False

    def __enter__(self) -> RealtimeTerminal:
        self._draw()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def update(self, *, status: str | None = None, **values: str) -> None:
        if status is not None:
            self._status = status
        for label, value in values.items():
            self._fields[label].value = value
        self._draw()

    def commit(self, label: str, text: str) -> None:
        self._clear()
        self._write_wrapped(f"✓ {self._display_label(label)}: ", text, color=GREEN)
        self._fields[label].value = ""
        self._draw()

    def error(self, message: str) -> None:
        self._clear()
        self._write_wrapped("Error: ", message, color=RED)
        self._closed = True

    def close(self, *, status: str | None = None) -> None:
        if self._closed:
            return
        if status is not None:
            self._status = status
        if self._interactive:
            self._draw()
            self._stream.write("\n")
        else:
            self._write_static()
        self._stream.flush()
        self._closed = True

    def _draw(self) -> None:
        if not self._interactive or self._closed:
            return
        self._clear()
        lines = self._panel_lines()
        for line in lines:
            self._stream.write(f"\x1b[2K{line}\n")
        self._stream.flush()
        self._rendered_lines = len(lines)

    def _clear(self) -> None:
        if not self._interactive or self._rendered_lines == 0:
            return
        self._stream.write(f"\x1b[{self._rendered_lines}F")
        for index in range(self._rendered_lines):
            self._stream.write("\x1b[2K")
            if index < self._rendered_lines - 1:
                self._stream.write("\x1b[1E")
        if self._rendered_lines > 1:
            self._stream.write(f"\x1b[{self._rendered_lines - 1}F")
        self._rendered_lines = 0

    def _panel_lines(self) -> list[str]:
        terminal = shutil.get_terminal_size(fallback=(80, 24))
        width = max(20, min(terminal.columns - 1, 100))
        fixed_lines = 4 + len(self._fields)
        available = max(len(self._fields), terminal.lines - fixed_lines - 1)
        lines_per_field = max(1, available // len(self._fields))

        lines = [
            self._style(self._title.upper(), BOLD, CYAN),
            self._style("─" * width, DIM),
        ]
        for field in self._fields.values():
            lines.append(self._style(self._display_label(field.label), BOLD))
            wrapped = self._wrap(field.value or "Waiting for speech…", width - 2)
            if len(wrapped) > lines_per_field:
                wrapped = wrapped[-lines_per_field:]
                wrapped[0] = f"…{wrapped[0][1:]}"
            lines.extend(f"  {line}" for line in wrapped)
        lines.extend(
            [
                self._style("─" * width, DIM),
                self._style(f"● {self._status}  ·  Ctrl+C to stop", DIM),
            ]
        )
        return lines

    def _write_static(self) -> None:
        self._stream.write(f"{self._title}\n")
        for field in self._fields.values():
            label = self._display_label(field.label)
            self._write_wrapped(f"{label}: ", field.value or "—")
        self._stream.write(f"Status: {self._status}\n")

    def _write_wrapped(self, prefix: str, text: str, *, color: str = "") -> None:
        width = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
        lines = self._wrap(text, max(1, width - len(prefix))) or [""]
        styled_prefix = self._style(prefix, BOLD, color)
        self._stream.write(f"{styled_prefix}{lines[0]}\n")
        indent = " " * len(prefix)
        for line in lines[1:]:
            self._stream.write(f"{indent}{line}\n")
        self._stream.flush()

    @staticmethod
    def _wrap(value: str, width: int) -> list[str]:
        return textwrap.wrap(
            " ".join(value.split()),
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )

    def _style(self, text: str, *styles: str) -> str:
        if not self._color:
            return text
        return f"{''.join(styles)}{text}{RESET}"

    @staticmethod
    def _display_label(label: str) -> str:
        return label.replace("_", " ").title()
