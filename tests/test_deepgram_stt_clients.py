import asyncio
import builtins
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from vocalbin.deepgram import (
    SpeechToText,
    SpeechToTextConfig,
    SpeechToTextEncoding,
    SpeechToTextError,
    SpeechToTextModel,
    StreamingSpeechToText,
    StreamingSpeechToTextConfig,
    StreamingSpeechToTextEncoding,
)
from vocalbin.deepgram import events as deepgram_events
from vocalbin.deepgram.stt import streaming as deepgram_streaming


class Word(BaseModel):
    word: str
    confidence: float
    start: float = 0.0


class ConnectedMessage(BaseModel):
    type: str = "Connected"
    request_id: str
    sequence_id: int = 0


class TurnInfoMessage(BaseModel):
    type: str = "TurnInfo"
    request_id: str
    sequence_id: int = 0
    event: str
    turn_index: int = 0
    audio_window_start: float = 0.0
    audio_window_end: float = 1.0
    transcript: str = ""
    words: list[Word] = []
    end_of_turn_confidence: float = 0.5


class FatalErrorMessage(BaseModel):
    type: str = "FatalError"
    sequence_id: int = 0
    code: str
    description: str


class FakeConnection:
    def __init__(
        self, messages: list[Any], send_error: Exception | None = None
    ) -> None:
        self.messages = messages
        self.send_error = send_error
        self.audio: list[bytes] = []
        self.controls: list[Any] = []
        self.closed = False

    async def send_media(self, chunk: bytes) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.audio.append(chunk)

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


class FakeListenV2:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.calls: list[dict[str, Any]] = []

    def connect(self, **kwargs: Any) -> FakeManager:
        self.calls.append(kwargs)
        return FakeManager(self.connection)


class FakeMedia:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def transcribe_file(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


class FakeStreamingClient:
    def __init__(self, connection: FakeConnection) -> None:
        self.v2 = FakeListenV2(connection)
        self.listen = SimpleNamespace(v2=self.v2)


class FakeRestClient:
    def __init__(self, result: Any) -> None:
        self.media = FakeMedia(result)
        self.listen = SimpleNamespace(v1=SimpleNamespace(media=self.media))


def transcription(
    transcript: str | None = "hello world",
    *,
    channels: list[Any] | None = None,
) -> Any:
    alternatives = [SimpleNamespace(transcript=transcript, confidence=0.98)]
    resolved_channels = (
        channels
        if channels is not None
        else [SimpleNamespace(alternatives=alternatives, detected_language="en")]
    )
    return SimpleNamespace(
        results=SimpleNamespace(channels=resolved_channels),
        metadata=SimpleNamespace(request_id="request-id"),
        model_dump=lambda mode: {"results": {"channels": "..."}},
    )


async def audio_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


@pytest.fixture(autouse=True)
def close_stream_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepgram_streaming, "_close_stream_message", lambda: {"type": "CloseStream"}
    )


async def test_deepgram_stt_transcribes_bytes_with_defaults() -> None:
    client = FakeRestClient(transcription())
    service = SpeechToText(client=cast(Any, client))

    response = await service.transcribe(b"audio")

    assert response.text == "hello world"
    assert response.model == SpeechToTextModel.NOVA_3
    assert response.request_id == "request-id"
    assert response.confidence == 0.98
    assert response.detected_language == "en"
    assert response.raw == {"results": {"channels": "..."}}
    assert client.media.calls == [{"request": b"audio", "model": "nova-3"}]


async def test_deepgram_stt_transcribes_a_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "utterance.wav"
    audio_path.write_bytes(b"file-audio")
    client = FakeRestClient(transcription())
    service = SpeechToText(client=cast(Any, client))

    await service.transcribe(str(audio_path))

    assert client.media.calls[0]["request"] == b"file-audio"


async def test_deepgram_stt_rejects_missing_files(tmp_path: Path) -> None:
    service = SpeechToText(client=cast(Any, FakeRestClient(transcription())))

    with pytest.raises(ValueError, match="does not exist"):
        await service.transcribe(tmp_path / "missing.wav")


async def test_deepgram_stt_rejects_empty_audio() -> None:
    service = SpeechToText(client=cast(Any, FakeRestClient(transcription())))

    with pytest.raises(ValueError, match="audio must not be empty"):
        await service.transcribe(b"")


async def test_deepgram_stt_accepts_flat_parameters() -> None:
    client = FakeRestClient(transcription())
    service = SpeechToText(client=cast(Any, client))

    await service.transcribe(
        b"audio",
        model=SpeechToTextModel.NOVA_3_MEDICAL,
        language="de",
        encoding=SpeechToTextEncoding.LINEAR16,
        keyterms=["vocalbin"],
        smart_format=True,
        diarize=False,
    )

    assert client.media.calls[0] == {
        "request": b"audio",
        "model": "nova-3-medical",
        "language": "de",
        "encoding": "linear16",
        "keyterm": ["vocalbin"],
        "smart_format": True,
        "diarize": False,
    }


async def test_deepgram_stt_uses_constructor_defaults() -> None:
    client = FakeRestClient(transcription())
    service = SpeechToText(client=cast(Any, client), punctuate=True, numerals=True)

    await service.transcribe(b"audio", config=SpeechToTextConfig(paragraphs=True))
    await service.transcribe(b"audio")

    assert "punctuate" not in client.media.calls[0]
    assert client.media.calls[0]["paragraphs"] is True
    assert client.media.calls[1]["punctuate"] is True
    assert client.media.calls[1]["numerals"] is True


async def test_deepgram_stt_rejects_config_with_flat_parameters() -> None:
    service = SpeechToText(client=cast(Any, FakeRestClient(transcription())))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await service.transcribe(b"audio", language="de", config=SpeechToTextConfig())


async def test_deepgram_stt_handles_responses_without_alternatives() -> None:
    empty_channel = SimpleNamespace(alternatives=[], detected_language=None)
    service = SpeechToText(
        client=cast(Any, FakeRestClient(transcription(channels=[empty_channel])))
    )

    assert (await service.transcribe(b"audio")).text == ""


async def test_deepgram_stt_handles_responses_without_channels() -> None:
    service = SpeechToText(client=cast(Any, FakeRestClient(transcription(channels=[]))))
    response = await service.transcribe(b"audio")

    assert response.text == ""
    assert response.detected_language is None


async def test_deepgram_stt_handles_alternatives_without_transcript() -> None:
    service = SpeechToText(client=cast(Any, FakeRestClient(transcription(None))))

    assert (await service.transcribe(b"audio")).text == ""


async def test_deepgram_stt_rejects_callback_responses() -> None:
    accepted = SimpleNamespace(
        request_id="request-id", model_dump=lambda mode: {"request_id": "request-id"}
    )
    service = SpeechToText(client=cast(Any, FakeRestClient(accepted)))

    with pytest.raises(ValueError, match="asynchronous callback response"):
        await service.transcribe(b"audio")


async def test_deepgram_streaming_stt_yields_normalized_turn_events() -> None:
    connection = FakeConnection(
        [
            ConnectedMessage(request_id="id"),
            TurnInfoMessage(request_id="id", event="StartOfTurn", transcript="Hi"),
            TurnInfoMessage(request_id="id", event="Update", transcript="Hi there"),
            TurnInfoMessage(request_id="id", event="EagerEndOfTurn"),
            TurnInfoMessage(request_id="id", event="TurnResumed"),
            TurnInfoMessage(
                request_id="id",
                event="EndOfTurn",
                transcript="Hi there",
                words=[Word(word="Hi", confidence=0.9)],
            ),
        ]
    )
    client = FakeStreamingClient(connection)
    service = StreamingSpeechToText(client=cast(Any, client))

    events = await collect(service.stream(audio_stream(b"", b"one", b"two")))

    assert [type(event) for event in events] == [
        deepgram_events.Connected,
        deepgram_events.TurnStart,
        deepgram_events.TurnUpdate,
        deepgram_events.TurnEagerEnd,
        deepgram_events.TurnResume,
        deepgram_events.TurnEnd,
    ]
    assert events[-1].words == [deepgram_events.Word(word="Hi", confidence=0.9)]
    assert connection.audio == [b"one", b"two"]
    assert connection.controls == [{"type": "CloseStream"}]
    assert connection.closed
    assert client.v2.calls == [
        {"model": "flux-general-en", "encoding": "linear16", "sample_rate": "16000"}
    ]


async def test_deepgram_streaming_stt_sends_query_parameters_as_strings() -> None:
    connection = FakeConnection([])
    client = FakeStreamingClient(connection)
    service = StreamingSpeechToText(client=cast(Any, client))

    await collect(
        service.stream(
            audio_stream(b"audio"),
            sample_rate=48000,
            keyterms=["vocalbin"],
            eot_threshold=0.8,
            eager_eot_threshold=0.6,
            eot_timeout_ms=7000,
        )
    )

    assert client.v2.calls[0] == {
        "model": "flux-general-en",
        "encoding": "linear16",
        "sample_rate": "48000",
        "keyterm": ["vocalbin"],
        "eot_threshold": "0.8",
        "eager_eot_threshold": "0.6",
        "eot_timeout_ms": "7000",
    }


async def test_deepgram_streaming_stt_uses_constructor_defaults() -> None:
    client = FakeStreamingClient(FakeConnection([]))
    service = StreamingSpeechToText(
        client=cast(Any, client),
        model="flux-general-multi",
        encoding=StreamingSpeechToTextEncoding.MULAW,
        sample_rate=8000,
    )

    await collect(service.stream(audio_stream(b"audio")))

    assert client.v2.calls[0]["model"] == "flux-general-multi"
    assert client.v2.calls[0]["encoding"] == "mulaw"
    assert client.v2.calls[0]["sample_rate"] == "8000"


async def test_deepgram_streaming_stt_uses_call_config() -> None:
    client = FakeStreamingClient(FakeConnection([]))
    service = StreamingSpeechToText(client=cast(Any, client))

    await collect(
        service.stream(
            audio_stream(b"audio"),
            config=StreamingSpeechToTextConfig(eot_timeout_ms=1000),
        )
    )

    assert client.v2.calls[0]["eot_timeout_ms"] == "1000"


async def test_deepgram_streaming_stt_rejects_config_with_flat_parameters() -> None:
    service = StreamingSpeechToText(
        client=cast(Any, FakeStreamingClient(FakeConnection([])))
    )

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await collect(
            service.stream(
                audio_stream(b"audio"),
                sample_rate=48000,
                config=StreamingSpeechToTextConfig(),
            )
        )


async def test_deepgram_streaming_stt_websocket_can_be_prewarmed() -> None:
    client = FakeStreamingClient(FakeConnection([]))
    service = StreamingSpeechToText(client=cast(Any, client))

    assert service.is_connected is False
    await service.connect()
    await service.connect()
    assert service.is_connected is True
    assert len(client.v2.calls) == 1

    await collect(service.stream(audio_stream(b"audio")))

    assert service.is_connected is False
    assert len(client.v2.calls) == 1


async def test_deepgram_streaming_stt_reconnects_for_a_different_config() -> None:
    client = FakeStreamingClient(FakeConnection([]))
    service = StreamingSpeechToText(client=cast(Any, client))

    await service.connect()
    await collect(service.stream(audio_stream(b"audio"), sample_rate=48000))

    assert [call["sample_rate"] for call in client.v2.calls] == ["16000", "48000"]


async def test_deepgram_streaming_stt_raises_typed_provider_error() -> None:
    connection = FakeConnection(
        [FatalErrorMessage(code="INVALID_MODEL", description="Unknown model")]
    )
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    with pytest.raises(SpeechToTextError, match="Unknown model") as error:
        await collect(service.stream(audio_stream(b"audio")))

    assert error.value.error_code == "INVALID_MODEL"
    assert connection.closed


async def test_deepgram_streaming_stt_rejects_unknown_messages() -> None:
    connection = FakeConnection([ConnectedMessage(type="Metadata", request_id="id")])
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    with pytest.raises(ValueError, match="Unsupported Deepgram STT message type"):
        await collect(service.stream(audio_stream(b"audio")))


async def test_deepgram_streaming_stt_rejects_unknown_turn_events() -> None:
    connection = FakeConnection([TurnInfoMessage(request_id="id", event="Shrug")])
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    with pytest.raises(ValueError, match="Unsupported Deepgram STT turn event"):
        await collect(service.stream(audio_stream(b"audio")))


async def test_deepgram_streaming_stt_rejects_empty_audio() -> None:
    connection = FakeConnection([])
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    with pytest.raises(ValueError, match="at least one non-empty chunk"):
        await collect(service.stream(audio_stream(b"")))

    assert connection.controls == []
    assert connection.closed


async def test_deepgram_streaming_stt_propagates_sender_errors() -> None:
    connection = FakeConnection([], send_error=RuntimeError("socket failed"))
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    with pytest.raises(RuntimeError, match="socket failed"):
        await collect(service.stream(audio_stream(b"audio")))

    assert connection.closed


async def test_deepgram_streaming_stt_waits_for_events_after_sending_audio() -> None:
    class SlowConnection(FakeConnection):
        async def __aiter__(self) -> AsyncIterator[Any]:
            await asyncio.sleep(0.01)
            if False:
                yield

    connection = SlowConnection([])
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    assert await collect(service.stream(audio_stream(b"audio"))) == []


async def test_deepgram_streaming_stt_receives_while_audio_is_still_sending() -> None:
    async def slow_audio() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.01)
        yield b"audio"

    connection = FakeConnection([ConnectedMessage(request_id="id")])
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))

    events = await collect(service.stream(slow_audio()))

    assert isinstance(events[0], deepgram_events.Connected)


async def test_deepgram_streaming_stt_consumer_can_stop_early() -> None:
    connection = FakeConnection(
        [
            ConnectedMessage(request_id="id"),
            TurnInfoMessage(request_id="id", event="Update"),
        ]
    )
    service = StreamingSpeechToText(client=cast(Any, FakeStreamingClient(connection)))
    stream = service.stream(audio_stream(b"audio"))

    assert isinstance(await anext(stream), deepgram_events.Connected)
    await stream.aclose()

    assert connection.closed


def test_deepgram_streaming_stt_close_stream_message_uses_official_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deepgram.extensions.types.sockets":
            return SimpleNamespace(ListenV2ControlMessage=lambda *, type: type)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert deepgram_streaming._close_stream_message() == "CloseStream"
