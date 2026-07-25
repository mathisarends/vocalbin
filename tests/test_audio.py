import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from vocalbin.realtime import AudioStreamInput, MicrophoneInput


async def chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


async def test_audio_stream_input_obeys_lifecycle() -> None:
    audio_input = AudioStreamInput(chunks(b"one", b"two"))

    assert audio_input.is_active is False
    await audio_input.start()
    assert [chunk async for chunk in audio_input.stream_chunks()] == [b"one", b"two"]

    await audio_input.stop()
    assert audio_input.is_active is False


async def test_audio_stream_input_stops_source_iteration() -> None:
    audio_input = AudioStreamInput(chunks(b"ignored"))
    await audio_input.start()
    await audio_input.stop()

    assert [chunk async for chunk in audio_input.stream_chunks()] == []


async def test_microphone_input_reports_missing_audio_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    microphone = MicrophoneInput()

    with pytest.raises(ImportError, match=r"vocalbin\[audio\]"):
        await microphone.start()


class FakeRawInputStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def read(self, chunk_size: int) -> tuple[bytearray, bool]:
        return bytearray(b"pcm"), False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


async def test_microphone_input_streams_and_closes_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams: list[FakeRawInputStream] = []

    def create_stream(**kwargs: Any) -> FakeRawInputStream:
        stream = FakeRawInputStream(**kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=create_stream),
    )
    microphone = MicrophoneInput(device_index=3, sample_rate=24000, chunk_size=4800)

    await microphone.start()
    await microphone.start()
    iterator = microphone.stream_chunks()
    assert await anext(iterator) == b"pcm"
    assert microphone.is_active is True
    assert streams[0].kwargs == {
        "samplerate": 24000,
        "blocksize": 4800,
        "channels": 1,
        "dtype": "int16",
        "device": 3,
    }

    await microphone.stop()
    await microphone.stop()
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert streams[0].started is True
    assert streams[0].stopped is True
    assert streams[0].closed is True
    assert microphone._read() is None


async def test_microphone_stream_stops_when_read_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microphone = MicrophoneInput()
    microphone._active = True
    microphone._stream = object()
    monkeypatch.setattr(microphone, "_read", lambda: None)

    assert [chunk async for chunk in microphone.stream_chunks()] == []

    microphone._stream = None
    assert [chunk async for chunk in microphone.stream_chunks()] == []
    await microphone.stop()
