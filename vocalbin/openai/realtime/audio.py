import asyncio
import importlib
import math
import threading
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from vocalbin.openai.realtime.ports import AudioInput


class AudioStreamInput(AudioInput):
    def __init__(self, source: AsyncIterable[bytes]) -> None:
        self._source = source
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            if not self._active:
                return
            yield chunk


class MicrophoneInput(AudioInput):
    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = 24000,
        chunk_size: int | None = None,
        *,
        chunk_duration_ms: float | None = None,
    ) -> None:
        if not isinstance(sample_rate, int) or sample_rate < 1:
            raise ValueError("sample_rate must be a positive integer")
        if chunk_size is not None and chunk_duration_ms is not None:
            raise ValueError("Pass either chunk_size or chunk_duration_ms, not both.")
        if chunk_duration_ms is not None:
            if not math.isfinite(chunk_duration_ms) or chunk_duration_ms <= 0:
                raise ValueError("chunk_duration_ms must be positive and finite")
            chunk_size = round(sample_rate * chunk_duration_ms / 1000)
        if chunk_size is None:
            chunk_size = 4800
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must represent at least one whole sample")
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._stream: Any | None = None
        self._active = False
        self._read_complete = threading.Event()
        self._read_complete.set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_duration_ms(self) -> float:
        return self._chunk_size * 1000 / self._sample_rate

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        if self._active:
            return

        try:
            sounddevice = importlib.import_module("sounddevice")
        except ImportError as exc:
            raise ImportError(
                "MicrophoneInput requires the 'audio' extra: "
                "pip install 'vocalbin[audio]'"
            ) from exc

        self._stream = sounddevice.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._chunk_size,
            channels=1,
            dtype="int16",
            device=self._device_index,
        )
        self._stream.start()
        self._active = True

    def _read(self) -> bytes | None:
        self._read_complete.clear()
        try:
            if not self._active or self._stream is None:
                return None
            data, _overflowed = self._stream.read(self._chunk_size)
            return bytes(data)
        finally:
            self._read_complete.set()

    async def stop(self) -> None:
        if not self._active:
            return

        self._active = False
        await asyncio.get_running_loop().run_in_executor(
            None,
            self._read_complete.wait,
            1.0,
        )
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        while self._active and self._stream is not None:
            chunk = await asyncio.get_running_loop().run_in_executor(None, self._read)
            if chunk is None:
                return
            yield chunk
