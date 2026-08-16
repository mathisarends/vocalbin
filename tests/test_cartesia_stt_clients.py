import asyncio
import builtins
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vocalbin.cartesia import (
    SpeechToText,
    SpeechToTextConfig,
    SpeechToTextEncoding,
    SpeechToTextError,
)
from vocalbin.cartesia import (
    events as cartesia_events,
)
from vocalbin.cartesia.stt import client as cartesia_stt_clients


class FakeConnection:
    def __init__(
        self,
        events: list[Any],
        send_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.send_error = send_error
        self.audio: list[bytes] = []
        self.commands: list[dict[str, str]] = []
        self.closed = False

    async def send_raw(self, chunk: bytes) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.audio.append(chunk)

    async def send(self, command: dict[str, str]) -> None:
        self.commands.append(command)

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self.events:
            await asyncio.sleep(0)
            yield event


class FakeManager:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeConnection:
        self.entered = True
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        self.exited = True
        self.connection.closed = True


class FakeAutoFinalize:
    def __init__(self, connection: FakeConnection) -> None:
        self.manager = FakeManager(connection)
        self.calls: list[dict[str, Any]] = []

    def websocket(self, **kwargs: Any) -> FakeManager:
        self.calls.append(kwargs)
        return self.manager


class FakeClient:
    def __init__(self, connection: FakeConnection) -> None:
        self.auto_finalize = FakeAutoFinalize(connection)
        self.stt = SimpleNamespace(auto_finalize=self.auto_finalize)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def event(type_: str, **values: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **values)


async def audio_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def collect(
    stream: AsyncIterator[Any],
) -> list[Any]:
    return [event async for event in stream]


async def test_cartesia_stt_stream_yields_normalized_turn_events() -> None:
    connection = FakeConnection(
        [
            event("connected", request_id="id"),
            event("turn.start", request_id="id"),
            event("turn.update", request_id="id", transcript="Hello"),
            event("turn.eager_end", request_id="id", transcript="Hello there"),
            event("turn.resume", request_id="id"),
            event("turn.end", request_id="id", transcript="Hello there"),
        ]
    )
    fake_client = FakeClient(connection)
    service = SpeechToText(client=cast(Any, fake_client))

    events = await collect(service.stream(audio_stream(b"", b"one", b"two")))

    assert [type(item) for item in events] == [
        cartesia_events.Connected,
        cartesia_events.TurnStart,
        cartesia_events.TurnUpdate,
        cartesia_events.TurnEagerEnd,
        cartesia_events.TurnResume,
        cartesia_events.TurnEnd,
    ]
    assert connection.audio == [b"one", b"two"]
    assert connection.commands == [{"type": "close"}]
    assert connection.closed
    assert fake_client.auto_finalize.calls == [
        {
            "model": "ink-2",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
        }
    ]


async def test_cartesia_stt_stream_uses_call_config() -> None:
    connection = FakeConnection([])
    fake_client = FakeClient(connection)
    service = SpeechToText(client=cast(Any, fake_client))
    config = SpeechToTextConfig(
        keyterms=["vocalbin"],
        turn_end_timeout_ms=640,
    )

    assert await collect(service.stream(audio_stream(b"audio"), config=config)) == []
    assert fake_client.auto_finalize.calls[0]["keyterm"] == ["vocalbin"]
    assert fake_client.auto_finalize.calls[0]["turn_end_timeout_ms"] == 640


async def test_cartesia_stt_stream_accepts_flat_parameters() -> None:
    connection = FakeConnection([])
    fake_client = FakeClient(connection)
    service = SpeechToText(client=cast(Any, fake_client))

    assert (
        await collect(
            service.stream(
                audio_stream(b"audio"),
                sample_rate=48000,
                keyterms=["vocalbin"],
            )
        )
        == []
    )
    assert fake_client.auto_finalize.calls[0]["sample_rate"] == 48000
    assert fake_client.auto_finalize.calls[0]["keyterm"] == ["vocalbin"]


async def test_cartesia_stt_stream_accepts_flat_model_and_encoding() -> None:
    connection = FakeConnection([])
    fake_client = FakeClient(connection)
    service = SpeechToText(client=cast(Any, fake_client))

    assert (
        await collect(
            service.stream(
                audio_stream(b"audio"),
                model="ink-future",
                encoding=SpeechToTextEncoding.PCM_F32LE,
            )
        )
        == []
    )
    assert fake_client.auto_finalize.calls[0]["model"] == "ink-future"
    assert fake_client.auto_finalize.calls[0]["encoding"] == "pcm_f32le"
    assert fake_client.auto_finalize.calls[0]["sample_rate"] == 16000


async def test_cartesia_stt_stream_rejects_config_with_flat_parameters() -> None:
    service = SpeechToText(client=cast(Any, FakeClient(FakeConnection([]))))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await collect(
            service.stream(
                audio_stream(b"audio"),
                sample_rate=48000,
                config=SpeechToTextConfig(),
            )
        )


async def test_cartesia_stt_stream_waits_for_events_after_sending_audio() -> None:
    class SlowConnection(FakeConnection):
        async def __aiter__(self) -> AsyncIterator[Any]:
            await asyncio.sleep(0.01)
            if False:
                yield

    connection = SlowConnection([])
    service = SpeechToText(client=cast(Any, FakeClient(connection)))

    assert await collect(service.stream(audio_stream(b"audio"))) == []
    assert connection.commands == [{"type": "close"}]


async def test_cartesia_stt_stream_receives_while_audio_is_still_sending() -> None:
    async def slow_audio() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.01)
        yield b"audio"

    connection = FakeConnection([event("connected", request_id="id")])
    service = SpeechToText(client=cast(Any, FakeClient(connection)))

    events = await collect(service.stream(slow_audio()))

    assert len(events) == 1
    assert isinstance(events[0], cartesia_events.Connected)
    assert connection.closed


async def test_cartesia_stt_stream_raises_typed_provider_error() -> None:
    connection = FakeConnection(
        [
            event(
                "error",
                message="Invalid model",
                error_code="model_not_found",
                status_code=400,
                request_id="id",
                doc_url="https://docs.cartesia.ai",
            )
        ]
    )
    service = SpeechToText(client=cast(Any, FakeClient(connection)))

    with pytest.raises(SpeechToTextError, match="Invalid model") as error:
        await collect(service.stream(audio_stream(b"audio")))

    assert error.value.error_code == "model_not_found"
    assert error.value.status_code == 400
    assert error.value.request_id == "id"
    assert error.value.doc_url == "https://docs.cartesia.ai"
    assert connection.closed


async def test_cartesia_stt_stream_rejects_empty_audio() -> None:
    connection = FakeConnection([])
    service = SpeechToText(client=cast(Any, FakeClient(connection)))

    with pytest.raises(ValueError, match="at least one non-empty chunk"):
        await collect(service.stream(audio_stream(b"")))

    assert connection.commands == []
    assert connection.closed


async def test_cartesia_stt_stream_propagates_sender_errors() -> None:
    connection = FakeConnection([], send_error=RuntimeError("socket failed"))
    service = SpeechToText(client=cast(Any, FakeClient(connection)))

    with pytest.raises(RuntimeError, match="socket failed"):
        await collect(service.stream(audio_stream(b"audio")))

    assert connection.closed


async def test_cartesia_stt_consumer_can_stop_early() -> None:
    connection = FakeConnection(
        [
            event("connected", request_id="id"),
            event("turn.start", request_id="id"),
        ]
    )
    service = SpeechToText(client=cast(Any, FakeClient(connection)))
    stream = service.stream(audio_stream(b"audio"))

    assert isinstance(await anext(stream), cartesia_events.Connected)
    await stream.aclose()

    assert connection.closed


async def test_cartesia_stt_context_manager_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection([])
    fake_client = FakeClient(connection)
    monkeypatch.setenv("CARTESIA_API_KEY", "environment-key")

    def create_client(api_key: str) -> Any:
        assert api_key == "environment-key"
        return fake_client

    monkeypatch.setattr(cartesia_stt_clients, "_create_client", create_client)

    async with SpeechToText() as service:
        await collect(service.stream(audio_stream(b"audio")))

    assert fake_client.closed


async def test_cartesia_stt_injected_client_is_not_closed() -> None:
    fake_client = FakeClient(FakeConnection([]))
    service = SpeechToText(client=cast(Any, fake_client))

    await service.aclose()

    assert not fake_client.closed


def test_cartesia_stt_api_key_and_client_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either 'api_key' or 'client'"):
        SpeechToText(
            api_key="key",
            client=cast(Any, FakeClient(FakeConnection([]))),
        )


def test_cartesia_stt_create_client_uses_official_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cartesia":
            return SimpleNamespace(
                AsyncCartesia=lambda *, api_key: SimpleNamespace(api_key=api_key)
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = cartesia_stt_clients._create_client("key")

    assert client.api_key == "key"


def test_cartesia_stt_create_client_explains_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cartesia":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"vocalbin\[cartesia\]"):
        cartesia_stt_clients._create_client("key")


async def test_cartesia_stt_stop_task_cancels_pending_task() -> None:
    task = asyncio.create_task(asyncio.sleep(60))

    await cartesia_stt_clients._stop_task(task)

    assert task.cancelled()


def test_cartesia_stt_rejects_unknown_provider_event() -> None:
    with pytest.raises(ValueError, match="Unsupported Cartesia STT event type"):
        cartesia_stt_clients._normalize_event(cast(Any, event("future.event")))
