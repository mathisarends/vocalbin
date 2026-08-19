import asyncio
import builtins
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from vocalbin.deepgram import (
    AudioContainer,
    AudioEncoding,
    TextToSpeech,
    TextToSpeechConfig,
    TextToSpeechError,
    TextToSpeechModel,
)
from vocalbin.deepgram import shared as deepgram_shared
from vocalbin.deepgram.tts import client as deepgram_tts_client


class MetadataMessage(BaseModel):
    type: str = "Metadata"
    request_id: str = "request-id"


class ControlMessage(BaseModel):
    type: str


class WarningMessage(BaseModel):
    type: str = "Warning"
    warn_msg: str
    warn_code: str


class FakeConnection:
    def __init__(
        self, messages: list[Any], send_error: Exception | None = None
    ) -> None:
        self.messages = messages
        self.send_error = send_error
        self.texts: list[Any] = []
        self.controls: list[Any] = []
        self.closed = False

    async def send_text(self, message: Any) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.texts.append(message)

    async def send_control(self, message: Any) -> None:
        self.controls.append(message)

    async def __aiter__(self) -> AsyncIterator[Any]:
        for message in self.messages:
            await asyncio.sleep(0)
            yield message


class FakeManager:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        self.connection.closed = True


class FakeSpeakV1:
    def __init__(self, connection: FakeConnection, chunks: list[bytes]) -> None:
        self.connection = connection
        self.calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.audio = SimpleNamespace(generate=self._generate)

    def connect(self, **kwargs: Any) -> FakeManager:
        self.calls.append(kwargs)
        return FakeManager(self.connection)

    async def _generate(self, **kwargs: Any) -> AsyncIterator[bytes]:
        self.generate_calls.append(kwargs)
        for chunk in (b"au", b"dio"):
            yield chunk


class FakeClient:
    def __init__(self, connection: FakeConnection | None = None) -> None:
        self.v1 = FakeSpeakV1(connection or FakeConnection([]), [])
        self.speak = SimpleNamespace(v1=self.v1)


async def text_stream(*chunks: str) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


@pytest.fixture(autouse=True)
def socket_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepgram_tts_client,
        "_text_message",
        lambda text: {"type": "Speak", "text": text},
    )
    monkeypatch.setattr(
        deepgram_tts_client, "_flush_message", lambda: {"type": "Flush"}
    )


async def test_deepgram_tts_generates_audio_with_defaults() -> None:
    client = FakeClient()
    service = TextToSpeech(client=cast(Any, client))

    response = await service.generate("Hallo aus vocalbin!")

    assert response.audio == b"audio"
    assert response.model == TextToSpeechModel.AURA_2_THALIA_EN
    assert response.encoding == AudioEncoding.LINEAR16
    assert response.container is None
    assert response.sample_rate == 24000
    assert response.content_type == "audio/pcm"
    assert client.v1.generate_calls == [
        {
            "text": "Hallo aus vocalbin!",
            "model": "aura-2-thalia-en",
            "encoding": "linear16",
            "sample_rate": 24000,
        }
    ]


async def test_deepgram_tts_generate_accepts_flat_parameters() -> None:
    client = FakeClient()
    service = TextToSpeech(client=cast(Any, client))

    response = await service.generate(
        "Hallo",
        model="aura-2-hera-en",
        encoding=AudioEncoding.MP3,
        bit_rate=48000,
        sample_rate=48000,
    )

    assert response.content_type == "audio/mpeg"
    assert client.v1.generate_calls[0] == {
        "text": "Hallo",
        "model": "aura-2-hera-en",
        "encoding": "mp3",
        "sample_rate": 48000,
        "bit_rate": 48000,
    }


async def test_deepgram_tts_generate_uses_constructor_defaults() -> None:
    client = FakeClient()
    service = TextToSpeech(
        client=cast(Any, client),
        container=AudioContainer.WAV,
        sample_rate=16000,
    )

    response = await service.generate("Hallo")

    assert response.content_type == "audio/wav"
    assert client.v1.generate_calls[0]["container"] == "wav"
    assert client.v1.generate_calls[0]["sample_rate"] == 16000


async def test_deepgram_tts_generate_uses_call_config() -> None:
    client = FakeClient()
    service = TextToSpeech(client=cast(Any, client))

    await service.generate(
        "Hallo", config=TextToSpeechConfig(model="aura-2-luna-en", sample_rate=None)
    )

    assert client.v1.generate_calls[0] == {
        "text": "Hallo",
        "model": "aura-2-luna-en",
        "encoding": "linear16",
    }


async def test_deepgram_tts_generate_rejects_config_with_flat_parameters() -> None:
    service = TextToSpeech(client=cast(Any, FakeClient()))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await service.generate(
            "Hallo", encoding=AudioEncoding.MP3, config=TextToSpeechConfig()
        )


async def test_deepgram_tts_generate_rejects_blank_text() -> None:
    service = TextToSpeech(client=cast(Any, FakeClient()))

    with pytest.raises(ValueError, match="text must not be blank"):
        await service.generate("   ")


async def test_deepgram_tts_generate_rejects_long_text() -> None:
    service = TextToSpeech(client=cast(Any, FakeClient()))

    with pytest.raises(ValueError, match="must not exceed 2000 characters"):
        await service.generate("a" * 2001)


async def test_deepgram_tts_streams_audio_chunks() -> None:
    connection = FakeConnection(
        [MetadataMessage(), b"au", b"dio", ControlMessage(type="Flushed")]
    )
    client = FakeClient(connection)
    service = TextToSpeech(client=cast(Any, client))

    chunks = await collect(service.stream("Hallo aus vocalbin!"))

    assert chunks == [b"au", b"dio"]
    assert connection.texts == [{"type": "Speak", "text": "Hallo aus vocalbin!"}]
    assert connection.controls == [{"type": "Flush"}]
    assert connection.closed
    assert client.v1.calls == [
        {"model": "aura-2-thalia-en", "encoding": "linear16", "sample_rate": "24000"}
    ]


async def test_deepgram_tts_streams_incremental_text() -> None:
    connection = FakeConnection([b"audio", ControlMessage(type="Flushed")])
    client = FakeClient(connection)
    service = TextToSpeech(client=cast(Any, client))

    chunks = await collect(
        service.stream_incremental(
            text_stream("Dieser Text ", "", "wird gestreamt."),
            model="aura-2-orion-en",
            sample_rate=16000,
        )
    )

    assert chunks == [b"audio"]
    assert connection.texts == [
        {"type": "Speak", "text": "Dieser Text "},
        {"type": "Speak", "text": "wird gestreamt."},
    ]
    assert client.v1.calls[0]["model"] == "aura-2-orion-en"
    assert client.v1.calls[0]["sample_rate"] == "16000"


async def test_deepgram_tts_stream_stops_when_the_socket_closes() -> None:
    connection = FakeConnection([b"audio"])
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))

    assert await collect(service.stream("Hallo")) == [b"audio"]


async def test_deepgram_tts_stream_raises_typed_provider_error() -> None:
    connection = FakeConnection(
        [WarningMessage(warn_msg="Text is too long", warn_code="TEXT_TOO_LONG")]
    )
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))

    with pytest.raises(TextToSpeechError, match="Text is too long") as error:
        await collect(service.stream("Hallo"))

    assert error.value.error_code == "TEXT_TOO_LONG"
    assert connection.closed


async def test_deepgram_tts_stream_rejects_compressed_encodings() -> None:
    service = TextToSpeech(client=cast(Any, FakeClient()), encoding=AudioEncoding.MP3)

    with pytest.raises(ValueError, match="linear16, mulaw, or alaw"):
        await collect(service.stream("Hallo"))


async def test_deepgram_tts_stream_rejects_containers() -> None:
    service = TextToSpeech(client=cast(Any, FakeClient()), container=AudioContainer.WAV)

    with pytest.raises(ValueError, match="requires no container"):
        await collect(service.stream("Hallo"))


async def test_deepgram_tts_stream_accepts_the_none_container() -> None:
    connection = FakeConnection([b"audio", ControlMessage(type="Flushed")])
    service = TextToSpeech(
        client=cast(Any, FakeClient(connection)), container=AudioContainer.NONE
    )

    assert await collect(service.stream("Hallo")) == [b"audio"]


async def test_deepgram_tts_stream_rejects_empty_text() -> None:
    connection = FakeConnection([])
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))

    with pytest.raises(ValueError, match="at least one non-empty chunk"):
        await collect(service.stream_incremental(text_stream("")))

    assert connection.controls == []


async def test_deepgram_tts_stream_propagates_sender_errors() -> None:
    connection = FakeConnection([], send_error=RuntimeError("socket failed"))
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))

    with pytest.raises(RuntimeError, match="socket failed"):
        await collect(service.stream("Hallo"))

    assert connection.closed


async def test_deepgram_tts_stream_waits_for_audio_after_sending_text() -> None:
    class SlowConnection(FakeConnection):
        async def __aiter__(self) -> AsyncIterator[Any]:
            await asyncio.sleep(0.01)
            if False:
                yield

    connection = SlowConnection([])
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))

    assert await collect(service.stream("Hallo")) == []


async def test_deepgram_tts_stream_receives_while_text_is_still_sending() -> None:
    async def slow_text() -> AsyncIterator[str]:
        await asyncio.sleep(0.01)
        yield "Hallo"

    connection = FakeConnection([b"audio", ControlMessage(type="Flushed")])
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))

    assert await collect(service.stream_incremental(slow_text())) == [b"audio"]


async def test_deepgram_tts_consumer_can_stop_early() -> None:
    connection = FakeConnection([b"au", b"dio", ControlMessage(type="Flushed")])
    service = TextToSpeech(client=cast(Any, FakeClient(connection)))
    stream = service.stream("Hallo")

    assert await anext(stream) == b"au"
    await stream.aclose()

    assert connection.closed


async def test_deepgram_tts_websocket_can_be_prewarmed() -> None:
    connection = FakeConnection([b"audio", ControlMessage(type="Flushed")])
    client = FakeClient(connection)
    service = TextToSpeech(client=cast(Any, client))

    assert service.is_connected is False
    await service.connect()
    await service.connect()
    assert service.is_connected is True

    await collect(service.stream("Hallo"))

    assert service.is_connected is False
    assert len(client.v1.calls) == 1


async def test_deepgram_tts_reconnects_for_a_different_stream_config() -> None:
    client = FakeClient(FakeConnection([]))
    service = TextToSpeech(client=cast(Any, client))

    await service.connect()
    await collect(service.stream("Hallo", sample_rate=16000))

    assert [call["sample_rate"] for call in client.v1.calls] == ["24000", "16000"]


async def test_deepgram_tts_context_manager_closes_the_owned_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_client = SimpleNamespace(closed=False)

    async def aclose() -> None:
        http_client.closed = True

    http_client.aclose = aclose
    monkeypatch.setenv("DEEPGRAM_API_KEY", "environment-key")

    def create_client(api_key: str) -> Any:
        assert api_key == "environment-key"
        return FakeClient(), http_client

    monkeypatch.setattr(deepgram_shared, "_create_client", create_client)

    async with TextToSpeech() as service:
        await service.generate("Hallo")

    assert http_client.closed


async def test_deepgram_injected_client_keeps_its_transport() -> None:
    service = TextToSpeech(client=cast(Any, FakeClient()))

    await service.aclose()

    assert service._http_client is None


def test_deepgram_api_key_and_client_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either 'api_key' or 'client'"):
        TextToSpeech(api_key="key", client=cast(Any, FakeClient()))


def test_deepgram_create_client_uses_official_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    http_client = object()

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deepgram":
            return SimpleNamespace(
                AsyncDeepgramClient=lambda *, api_key, httpx_client: SimpleNamespace(
                    api_key=api_key, httpx_client=httpx_client
                )
            )
        if name == "httpx":
            return SimpleNamespace(
                AsyncClient=lambda *, timeout, follow_redirects: http_client
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client, transport = deepgram_shared._create_client("key")

    assert client.api_key == "key"
    assert transport is http_client


def test_deepgram_create_client_explains_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deepgram":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"vocalbin\[deepgram\]"):
        deepgram_shared._create_client("key")


def test_deepgram_tts_socket_messages_use_the_official_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deepgram.extensions.types.sockets":
            return SimpleNamespace(
                SpeakV1TextMessage=lambda *, type, text: (type, text),
                SpeakV1ControlMessage=lambda *, type: type,
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert deepgram_tts_client._text_message("Hallo") == ("Speak", "Hallo")
    assert deepgram_tts_client._flush_message() == "Flush"


async def test_deepgram_tts_flat_parameters_keep_the_default_sample_rate() -> None:
    client = FakeClient()
    service = TextToSpeech(client=cast(Any, client))

    await service.generate("Hallo", encoding=AudioEncoding.FLAC)

    assert client.v1.generate_calls[0]["sample_rate"] == 24000
    assert client.v1.generate_calls[0]["model"] == "aura-2-thalia-en"
