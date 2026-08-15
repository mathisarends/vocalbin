import asyncio
import builtins
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vocalbin.cartesia import (
    CartesiaMp3OutputFormat,
    CartesiaRawOutputFormat,
    CartesiaTextToSpeech,
    CartesiaTextToSpeechConfig,
    CartesiaTextToSpeechError,
    CartesiaWavOutputFormat,
)
from vocalbin.cartesia import clients as cartesia_clients


class FakeBinaryResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def read(self) -> bytes:
        return self.content


class FakeContext:
    def __init__(self, events: list[Any], push_error: Exception | None = None) -> None:
        self.events = events
        self.push_error = push_error
        self.pushed: list[str] = []
        self.finished = False
        self.cancelled = False

    async def push(self, text: str) -> None:
        if self.push_error is not None:
            raise self.push_error
        self.pushed.append(text)

    async def no_more_inputs(self) -> None:
        self.finished = True

    async def receive(self) -> AsyncIterator[Any]:
        for event in self.events:
            await asyncio.sleep(0)
            yield event

    async def cancel(self) -> None:
        self.cancelled = True


class FakeConnection:
    def __init__(self, contexts: list[FakeContext]) -> None:
        self.contexts = contexts
        self.calls: list[dict[str, Any]] = []

    def context(self, **kwargs: Any) -> FakeContext:
        self.calls.append(kwargs)
        return self.contexts.pop(0)


class FakeConnectionManager:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> FakeConnection:
        self.entered += 1
        await asyncio.sleep(0)
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        self.exited += 1


class FakeTTS:
    def __init__(
        self,
        connection: FakeConnection,
        response: FakeBinaryResponse | None = None,
    ) -> None:
        self.connection = connection
        self.response = response or FakeBinaryResponse(b"generated")
        self.manager = FakeConnectionManager(connection)
        self.generate_calls: list[dict[str, Any]] = []
        self.websocket_calls = 0

    async def generate(self, **kwargs: Any) -> FakeBinaryResponse:
        self.generate_calls.append(kwargs)
        return self.response

    def websocket_connect(self) -> FakeConnectionManager:
        self.websocket_calls += 1
        return self.manager


class FakeClient:
    def __init__(self, contexts: list[FakeContext] | None = None) -> None:
        self.connection = FakeConnection(contexts or [])
        self.tts = FakeTTS(self.connection)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def event(type_: str, **values: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **values)


async def collect(stream: AsyncIterator[bytes]) -> list[bytes]:
    return [chunk async for chunk in stream]


async def text_stream(*chunks: str) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


@pytest.mark.parametrize(
    ("output_format", "content_type"),
    [
        (CartesiaRawOutputFormat(), "audio/pcm"),
        (CartesiaWavOutputFormat(), "audio/wav"),
        (CartesiaMp3OutputFormat(), "audio/mpeg"),
    ],
)
async def test_cartesia_generate_returns_normalized_response(
    output_format: CartesiaRawOutputFormat
    | CartesiaWavOutputFormat
    | CartesiaMp3OutputFormat,
    content_type: str,
) -> None:
    fake_client = FakeClient()
    service = CartesiaTextToSpeech(client=cast(Any, fake_client))

    response = await service.generate(
        "Hallo",
        voice_id="voice-id",
        model="sonic-custom",
        output_format=output_format,
        language="de",
        emotion="positivity:high",
        speed=1.1,
        volume=1.2,
        pronunciation_dict_id="dictionary-id",
    )

    assert response.audio == b"generated"
    assert response.content_type == content_type
    assert response.output_format == output_format
    assert fake_client.tts.generate_calls == [
        {
            "model_id": "sonic-custom",
            "voice": {"mode": "id", "id": "voice-id"},
            "output_format": output_format.model_dump(mode="json"),
            "language": "de",
            "generation_config": {
                "emotion": "positivity:high",
                "speed": 1.1,
                "volume": 1.2,
            },
            "pronunciation_dict_id": "dictionary-id",
            "transcript": "Hallo",
        }
    ]


async def test_cartesia_generate_accepts_legacy_config() -> None:
    fake_client = FakeClient()
    service = CartesiaTextToSpeech(client=cast(Any, fake_client))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id", language="de")

    response = await service.generate("Hallo", config=config)

    assert response.voice_id == "voice-id"
    assert fake_client.tts.generate_calls == [
        {**config.to_cartesia_params(), "transcript": "Hallo"}
    ]


async def test_cartesia_generate_rejects_config_with_flat_parameters() -> None:
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient()))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await cast(Any, service).generate(
            "Hallo",
            voice_id="voice-id",
            config=CartesiaTextToSpeechConfig(voice_id="other-voice"),
        )


async def test_cartesia_generate_requires_voice_id_for_flat_parameters() -> None:
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient()))

    with pytest.raises(ValueError, match="'voice_id' is required"):
        await cast(Any, service).generate("Hallo", language="de")


async def test_cartesia_generate_uses_default_config() -> None:
    fake_client = FakeClient()
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")
    service = CartesiaTextToSpeech(client=cast(Any, fake_client), default_config=config)

    response = await service.generate("Hallo")

    assert response.audio == b"generated"
    assert response.voice_id == "voice-id"


async def test_cartesia_generate_requires_config() -> None:
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient()))

    with pytest.raises(ValueError, match="Provide 'config'"):
        await service.generate("Hallo")


async def test_cartesia_generate_rejects_blank_text() -> None:
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient()))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    with pytest.raises(ValueError, match="text must not be blank"):
        await service.generate("   ", config=config)


async def test_cartesia_stream_yields_audio_and_reuses_websocket() -> None:
    first = FakeContext(
        [
            event("timestamps"),
            event("chunk", audio=b"one"),
            event("chunk", audio=None),
            event("done"),
        ]
    )
    second = FakeContext([event("chunk", audio=b"two"), event("done")])
    fake_client = FakeClient([first, second])
    service = CartesiaTextToSpeech(client=cast(Any, fake_client))

    assert await collect(
        service.stream(
            "Hallo",
            voice_id="voice-id",
            max_buffer_delay_ms=100,
            timeout=3,
        )
    ) == [b"one"]
    assert await collect(
        service.stream(
            "Hallo",
            voice_id="voice-id",
            max_buffer_delay_ms=100,
            timeout=3,
        )
    ) == [b"two"]

    assert first.pushed == ["Hallo"]
    assert first.finished
    assert fake_client.tts.websocket_calls == 1
    assert fake_client.connection.calls[0]["timeout"] == 3
    assert fake_client.connection.calls[0]["max_buffer_delay_ms"] == 100


async def test_cartesia_stream_incremental_accepts_async_text_chunks() -> None:
    context = FakeContext([event("chunk", audio=b"audio"), event("done")])
    fake_client = FakeClient([context])
    service = CartesiaTextToSpeech(client=cast(Any, fake_client))
    audio = await collect(
        service.stream_incremental(
            text_stream("", "Hallo ", "Welt"), voice_id="voice-id"
        )
    )

    assert audio == [b"audio"]
    assert context.pushed == ["Hallo ", "Welt"]
    assert context.finished


async def test_cartesia_stream_raises_typed_provider_error() -> None:
    context = FakeContext(
        [
            event(
                "error",
                message=None,
                title="Invalid model",
                error_code="model_not_found",
                status_code=400,
            )
        ]
    )
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    with pytest.raises(CartesiaTextToSpeechError, match="Invalid model") as error:
        await collect(service.stream("Hallo", config=config))

    assert error.value.error_code == "model_not_found"
    assert error.value.status_code == 400
    assert not context.cancelled


async def test_cartesia_stream_uses_default_error_message() -> None:
    context = FakeContext(
        [
            event(
                "error",
                message=None,
                title=None,
                error_code=None,
                status_code=None,
            )
        ]
    )
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    with pytest.raises(CartesiaTextToSpeechError, match="Cartesia TTS failed"):
        await collect(service.stream("Hallo", config=config))


async def test_cartesia_stream_cancels_context_when_consumer_stops() -> None:
    context = FakeContext([event("chunk", audio=b"audio"), event("done")])
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")
    stream = service.stream("Hallo", config=config)

    assert await anext(stream) == b"audio"
    await stream.aclose()

    assert context.cancelled


async def test_cartesia_stream_rejects_non_raw_output() -> None:
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient()))
    config = CartesiaTextToSpeechConfig(
        voice_id="voice-id",
        output_format=CartesiaWavOutputFormat(),
    )

    with pytest.raises(ValueError, match="requires raw output"):
        await collect(service.stream("Hallo", config=config))


async def test_cartesia_stream_rejects_empty_text_stream() -> None:
    context = FakeContext([])
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    with pytest.raises(ValueError, match="at least one non-empty chunk"):
        await collect(service.stream_incremental(text_stream(""), config=config))

    assert context.cancelled


async def test_cartesia_stream_cancels_when_response_ends_without_done() -> None:
    context = FakeContext([])
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    assert await collect(service.stream("Hallo", config=config)) == []
    assert context.cancelled


async def test_cartesia_stream_waits_for_response_after_sender_finishes() -> None:
    class SlowResponseContext(FakeContext):
        async def receive(self) -> AsyncIterator[Any]:
            await asyncio.sleep(0.01)
            yield event("done")

    context = SlowResponseContext([])
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    assert await collect(service.stream("Hallo", config=config)) == []
    assert context.finished


async def test_cartesia_stream_propagates_sender_errors() -> None:
    context = FakeContext([], push_error=RuntimeError("socket failed"))
    service = CartesiaTextToSpeech(client=cast(Any, FakeClient([context])))
    config = CartesiaTextToSpeechConfig(voice_id="voice-id")

    with pytest.raises(RuntimeError, match="socket failed"):
        await collect(service.stream("Hallo", config=config))

    assert context.cancelled


async def test_cartesia_connection_initialization_is_concurrency_safe() -> None:
    fake_client = FakeClient()
    service = CartesiaTextToSpeech(client=cast(Any, fake_client))

    first, second = await asyncio.gather(
        service._get_connection(), service._get_connection()
    )

    assert first is second
    assert fake_client.tts.websocket_calls == 1


async def test_cartesia_context_manager_closes_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = FakeContext([event("done")])
    fake_client = FakeClient([context])
    monkeypatch.setenv("CARTESIA_API_KEY", "environment-key")

    def create_client(api_key: str) -> Any:
        assert api_key == "environment-key"
        return fake_client

    monkeypatch.setattr(cartesia_clients, "_create_client", create_client)

    async with CartesiaTextToSpeech() as service:
        await collect(
            service.stream(
                "Hallo", config=CartesiaTextToSpeechConfig(voice_id="voice-id")
            )
        )

    assert fake_client.closed
    assert fake_client.tts.manager.exited == 1


async def test_cartesia_injected_client_is_not_closed() -> None:
    fake_client = FakeClient()
    service = CartesiaTextToSpeech(client=cast(Any, fake_client))

    await service.aclose()

    assert not fake_client.closed


def test_cartesia_api_key_and_client_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either 'api_key' or 'client'"):
        CartesiaTextToSpeech(api_key="key", client=cast(Any, FakeClient()))


def test_create_client_uses_official_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cartesia":
            return SimpleNamespace(
                AsyncCartesia=lambda *, api_key: SimpleNamespace(api_key=api_key)
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = cartesia_clients._create_client("key")

    assert client.api_key == "key"


def test_create_client_explains_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cartesia":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"vocalbin\[cartesia\]"):
        cartesia_clients._create_client("key")


async def test_stop_task_cancels_pending_task() -> None:
    task = asyncio.create_task(asyncio.sleep(60))

    await cartesia_clients._stop_task(task)

    assert task.cancelled()
