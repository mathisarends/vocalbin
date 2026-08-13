import asyncio
import base64
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import vocalbin.openai.realtime.clients as realtime_module
from vocalbin.openai.realtime import (
    AudioInput,
    AudioStreamInput,
    MicrophoneInput,
    OpenAIRealtimeProvider,
    OpenAIRealtimeTranscriber,
    OpenAIRealtimeTranslator,
    RealtimeError,
    RealtimeLogprob,
    RealtimeNoiseReduction,
    RealtimeProvider,
    RealtimeSessionConnected,
    RealtimeSessionType,
    RealtimeSourceTranscriptDelta,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationClosed,
    RealtimeTranslationConfig,
    RealtimeTranslationLanguage,
    RealtimeTranslationTranscriptDelta,
    SemanticVadConfig,
)
from vocalbin.openai.realtime.base import TRANSCRIPTION_SPEC, TRANSLATION_SPEC
from vocalbin.openai.realtime.messages import RealtimeTranscriptionAudioCommit


async def chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


class DummyProvider(RealtimeProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def build_url(
        self,
        session_type: RealtimeSessionType,
        model: str,
    ) -> str:
        self.calls.append((session_type, model))
        return f"ws://test/{session_type}"

    def build_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer test"}


class FakeConnection:
    def __init__(
        self,
        messages: list[object] | None = None,
        *,
        wait_until_closed: bool = False,
    ) -> None:
        self.messages = list(messages or [])
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._wait_until_closed = wait_until_closed
        self._closed = asyncio.Event()

    def __aiter__(self) -> "FakeConnection":
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0)
        if self.messages:
            return json.dumps(self.messages.pop(0))
        if self._wait_until_closed and not self.closed:
            await self._closed.wait()
        raise StopAsyncIteration

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


def install_connection(
    monkeypatch: pytest.MonkeyPatch,
    *connections: FakeConnection,
) -> None:
    remaining = list(connections)

    async def connect(url: str, headers: dict[str, str]) -> FakeConnection:
        assert url.startswith("ws://test/")
        assert headers == {"Authorization": "Bearer test"}
        return remaining.pop(0)

    monkeypatch.setattr(realtime_module, "_connect_websocket", connect)


def test_openai_realtime_provider_builds_urls_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIRealtimeProvider(
        "key",
        safety_identifier="hashed-user",
        base_url="wss://example.test/v1/realtime/",
    )

    assert provider.build_url(RealtimeSessionType.TRANSCRIPTION, "ignored") == (
        "wss://example.test/v1/realtime?intent=transcription"
    )
    assert provider.build_url(
        RealtimeSessionType.TRANSLATION, "gpt-realtime-translate"
    ) == ("wss://example.test/v1/realtime/translations?model=gpt-realtime-translate")
    assert provider.build_headers() == {
        "Authorization": "Bearer key",
        "OpenAI-Safety-Identifier": "hashed-user",
    }

    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    environment_provider = OpenAIRealtimeProvider()
    assert environment_provider.build_headers() == {
        "Authorization": "Bearer environment-key"
    }

    with pytest.raises(ValueError, match="must not be blank"):
        OpenAIRealtimeProvider("key", safety_identifier=" ")


async def test_connect_websocket_loads_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    expected = object()

    async def connect(url: str, additional_headers: dict[str, str]) -> object:
        calls.append((url, additional_headers))
        return expected

    def import_module(name: str) -> Any:
        assert name == "websockets.asyncio.client"
        return SimpleNamespace(connect=connect)

    monkeypatch.setattr(realtime_module.importlib, "import_module", import_module)

    result = await realtime_module._connect_websocket(
        "ws://test", {"Authorization": "Bearer key"}
    )

    assert result is expected
    assert calls == [("ws://test", {"Authorization": "Bearer key"})]


async def test_connect_websocket_explains_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_module(name: str) -> Any:
        assert name == "websockets.asyncio.client"
        raise ImportError

    monkeypatch.setattr(realtime_module.importlib, "import_module", import_module)

    with pytest.raises(ImportError, match=r"vocalbin\[realtime\]"):
        await realtime_module._connect_websocket("ws://test", {})


async def test_realtime_websocket_connects_sends_reads_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DummyProvider()
    first = FakeConnection()
    second = FakeConnection([{"type": "known"}])
    install_connection(monkeypatch, first, second)
    websocket = realtime_module._RealtimeWebSocket(
        provider,
        RealtimeSessionType.TRANSCRIPTION,
        "gpt-realtime-whisper",
    )

    assert websocket.is_connected is False
    await websocket.connect()
    await websocket.connect()
    await websocket.send(RealtimeTranscriptionAudioCommit())
    assert [event async for event in websocket.events()] == [{"type": "known"}]
    await websocket.close()
    await websocket.close()

    assert first.closed is True
    assert second.sent == [{"type": "input_audio_buffer.commit"}]
    assert second.closed is True
    assert provider.calls == [
        ("transcription", "gpt-realtime-whisper"),
        ("transcription", "gpt-realtime-whisper"),
    ]

    with pytest.raises(RuntimeError, match="not connected"):
        await websocket.send(RealtimeTranscriptionAudioCommit())
    with pytest.raises(RuntimeError, match="not connected"):
        await anext(websocket.events())


async def test_realtime_websocket_rejects_non_object_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection([["not", "an", "object"]])
    install_connection(monkeypatch, connection)
    websocket = realtime_module._RealtimeWebSocket(
        DummyProvider(),
        RealtimeSessionType.TRANSLATION,
        "gpt-realtime-translate",
    )
    await websocket.connect()

    with pytest.raises(ValueError, match="JSON object"):
        await anext(websocket.events())

    await websocket.close()


async def test_realtime_transcriber_streams_and_maps_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(
        [
            {"type": "session.created"},
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "event_id": "evt-1",
                "item_id": "item-1",
                "delta": "Hal",
                "logprobs": [{"token": "Hal", "logprob": -0.1}],
            },
            {
                "type": "input_audio_buffer.speech_started",
                "item_id": "item-1",
                "audio_start_ms": 20,
            },
            {
                "type": "input_audio_buffer.speech_stopped",
                "item_id": "item-1",
                "audio_end_ms": 420,
            },
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": "retry",
                    "code": "temporary",
                    "event_id": "client-1",
                    "param": "audio",
                },
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": "evt-2",
                "item_id": "item-1",
                "transcript": "Hallo",
                "logprobs": [],
                "usage": {"type": "duration", "seconds": 0.4},
            },
        ]
    )
    install_connection(monkeypatch, connection)
    service = OpenAIRealtimeTranscriber(
        RealtimeTranscriptionConfig(
            model="gpt-4o-transcribe",
            language="de",
            noise_reduction=None,
            turn_detection=SemanticVadConfig(eagerness="high"),
            include_logprobs=True,
        ),
        audio_input=AudioStreamInput(chunks(b"", b"pcm")),
        provider=DummyProvider(),
    )

    events = [event async for event in service.stream()]

    assert isinstance(events[0], RealtimeSessionConnected)
    assert isinstance(events[1], RealtimeTranscriptDelta)
    assert events[1].delta == "Hal"
    assert events[1].logprobs == [RealtimeLogprob(token="Hal", logprob=-0.1)]
    assert isinstance(events[2], RealtimeSpeechStarted)
    assert isinstance(events[3], RealtimeSpeechStopped)
    assert isinstance(events[4], RealtimeError)
    assert str(events[4].error) == "[server_error] retry"
    assert isinstance(events[5], RealtimeTranscriptCompleted)
    assert events[5].transcript == "Hallo"
    assert connection.sent == [
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": "gpt-4o-transcribe",
                            "language": "de",
                        },
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": "high",
                        },
                        "noise_reduction": None,
                    }
                },
                "include": ["item.input_audio_transcription.logprobs"],
            },
        },
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"pcm").decode("ascii"),
        },
        {"type": "input_audio_buffer.commit"},
    ]
    assert connection.closed is True


async def test_realtime_translator_streams_audio_and_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_audio = base64.b64encode(b"translated").decode("ascii")
    connection = FakeConnection(
        [
            {"type": "session.created"},
            {
                "type": "session.input_transcript.delta",
                "event_id": "evt-1",
                "delta": "hello",
                "elapsed_ms": 200,
            },
            {
                "type": "session.output_transcript.delta",
                "event_id": "evt-2",
                "delta": "hallo",
                "elapsed_ms": 400,
            },
            {
                "type": "session.output_audio.delta",
                "event_id": "evt-3",
                "delta": encoded_audio,
                "elapsed_ms": 400,
                "sample_rate": 16000,
                "channels": 2,
                "format": "pcm16",
            },
            {
                "type": "error",
                "error": {"type": "server_error", "message": "recoverable"},
            },
            {"type": "session.closed", "event_id": "evt-4"},
        ]
    )
    install_connection(monkeypatch, connection)
    service = OpenAIRealtimeTranslator(
        RealtimeTranslationConfig(
            target_language=RealtimeTranslationLanguage.GERMAN,
            noise_reduction=None,
            include_source_transcript=False,
        ),
        audio_input=AudioStreamInput(chunks(b"pcm")),
        provider=DummyProvider(),
    )

    events = [event async for event in service.stream()]

    assert isinstance(events[0], RealtimeSessionConnected)
    assert events[1] == RealtimeSourceTranscriptDelta(
        delta="hello", elapsed_ms=200, event_id="evt-1"
    )
    assert events[2] == RealtimeTranslationTranscriptDelta(
        delta="hallo", elapsed_ms=400, event_id="evt-2"
    )
    assert events[3] == RealtimeTranslationAudioDelta(
        audio=b"translated",
        elapsed_ms=400,
        sample_rate=16000,
        channels=2,
        event_id="evt-3",
    )
    assert isinstance(events[4], RealtimeError)
    assert events[5] == RealtimeTranslationClosed(event_id="evt-4")
    assert connection.sent == [
        {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "transcription": None,
                        "noise_reduction": None,
                    },
                    "output": {"language": "de"},
                }
            },
        },
        {
            "type": "session.input_audio_buffer.append",
            "audio": base64.b64encode(b"pcm").decode("ascii"),
        },
        {"type": "session.close"},
    ]


class BlockingAudioInput(AudioInput):
    def __init__(self) -> None:
        self.started = False
        self.stop_calls = 0
        self.wait = asyncio.Event()

    @property
    def is_active(self) -> bool:
        return self.started

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.started = False

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        await self.wait.wait()
        yield b"late"


async def test_transcriber_flush_and_lifecycle_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_start = OpenAIRealtimeTranscriber(
        audio_input=AudioStreamInput(chunks()),
        provider=DummyProvider(),
    )
    with pytest.raises(RuntimeError, match="before stream"):
        await before_start.flush()
    await before_start.stop()
    await before_start.stop()
    with pytest.raises(RuntimeError, match="already been stopped"):
        before_start.stream()

    connection = FakeConnection(wait_until_closed=True)
    install_connection(monkeypatch, connection)
    audio_input = BlockingAudioInput()
    service = OpenAIRealtimeTranscriber(
        audio_input=audio_input,
        provider=DummyProvider(),
    )
    stream = service.stream()
    assert isinstance(await anext(stream), RealtimeSessionConnected)

    with pytest.raises(RuntimeError, match="single-use"):
        service.stream()
    await service.flush()
    await service.stop()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    with pytest.raises(RuntimeError, match="has closed"):
        await service.flush()

    assert {"type": "input_audio_buffer.commit"} in connection.sent
    assert audio_input.stop_calls == 1


async def test_sender_errors_are_raised_by_event_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_source() -> AsyncIterator[bytes]:
        if False:
            yield b""
        raise RuntimeError("audio failed")

    connection = FakeConnection(wait_until_closed=True)
    install_connection(monkeypatch, connection)
    service = OpenAIRealtimeTranslator(
        RealtimeTranslationConfig(target_language=RealtimeTranslationLanguage.FRENCH),
        audio_input=AudioStreamInput(broken_source()),
        provider=DummyProvider(),
    )
    stream = service.stream()
    assert isinstance(await anext(stream), RealtimeSessionConnected)

    with pytest.raises(RuntimeError, match="audio failed"):
        await anext(stream)

    assert connection.closed is True


async def test_sender_returns_when_connection_is_not_open() -> None:
    service = OpenAIRealtimeTranscriber(
        audio_input=AudioStreamInput(chunks(b"ignored")),
        provider=DummyProvider(),
    )

    await service._audio_input.start()
    await service._send_audio()

    assert service._input_finished is False

    empty_service = OpenAIRealtimeTranscriber(
        audio_input=AudioStreamInput(chunks()),
        provider=DummyProvider(),
    )
    empty_service._stop_called = True
    await empty_service._audio_input.start()
    await empty_service._send_audio()

    assert empty_service._input_finished is True


async def test_sender_preserves_task_cancellation() -> None:
    async def cancelled_source() -> AsyncIterator[bytes]:
        if False:
            yield b""
        raise asyncio.CancelledError

    service = OpenAIRealtimeTranscriber(
        audio_input=AudioStreamInput(cancelled_source()),
        provider=DummyProvider(),
    )
    await service._audio_input.start()

    with pytest.raises(asyncio.CancelledError):
        await service._send_audio()


async def test_context_manager_and_default_realtime_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    transcriber = OpenAIRealtimeTranscriber(api_key="key")
    translator = OpenAIRealtimeTranslator(
        RealtimeTranslationConfig(
            target_language=RealtimeTranslationLanguage.SPANISH,
            noise_reduction=RealtimeNoiseReduction.NEAR_FIELD,
        ),
        api_key="key",
    )

    assert isinstance(transcriber._audio_input, MicrophoneInput)
    assert isinstance(translator._audio_input, MicrophoneInput)
    transcriber_update = transcriber._session_update.model_dump(mode="json")
    translator_update = translator._session_update.model_dump(mode="json")

    assert transcriber_update["session"]["audio"]["input"]["noise_reduction"] == {
        "type": "far_field"
    }
    assert translator_update["session"]["audio"] == {
        "input": {
            "transcription": {"model": "gpt-realtime-whisper"},
            "noise_reduction": {"type": "near_field"},
        },
        "output": {"language": "es"},
    }

    async with transcriber as entered:
        assert entered is transcriber
    await translator.stop()


def test_provider_and_credentials_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either 'provider'"):
        OpenAIRealtimeTranscriber(provider=DummyProvider(), api_key="key")

    with pytest.raises(ValueError, match="either 'provider'"):
        OpenAIRealtimeTranslator(
            RealtimeTranslationConfig(
                target_language=RealtimeTranslationLanguage.ENGLISH
            ),
            provider=DummyProvider(),
            safety_identifier="user",
        )


def test_event_mappers_preserve_defaults_and_ignore_unknown_events() -> None:
    audio = TRANSLATION_SPEC.parse_event(
        {
            "type": "session.output_audio.delta",
            "delta": base64.b64encode(b"audio").decode("ascii"),
        }
    )

    assert audio == RealtimeTranslationAudioDelta(audio=b"audio")
    assert TRANSLATION_SPEC.parse_event({"type": "unknown"}) is None
    assert TRANSCRIPTION_SPEC.parse_event({"type": "unknown"}) is None

    with pytest.raises(ValidationError):
        TRANSCRIPTION_SPEC.parse_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "missing item id",
            }
        )
