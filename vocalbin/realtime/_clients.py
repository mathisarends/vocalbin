import asyncio
import base64
import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import suppress
from types import TracebackType
from typing import Any, Literal, Self, cast
from urllib.parse import urlencode

from vocalbin.credentials import OpenAICredentials
from vocalbin.models import (
    RealtimeError,
    RealtimeErrorDetails,
    RealtimeSessionConnected,
    RealtimeSourceTranscriptDelta,
    RealtimeSpeechStarted,
    RealtimeSpeechStopped,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptDelta,
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionEvent,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationClosed,
    RealtimeTranslationConfig,
    RealtimeTranslationEvent,
    RealtimeTranslationTranscriptDelta,
)
from vocalbin.ports import (
    AudioInput,
    RealtimeProvider,
    RealtimeTranscription,
    RealtimeTranslation,
)
from vocalbin.realtime._audio import MicrophoneInput

type JsonObject = dict[str, Any]


async def _connect_websocket(url: str, headers: dict[str, str]) -> Any:
    try:
        connect = importlib.import_module("websockets.asyncio.client").connect
    except ImportError as exc:
        raise ImportError(
            "Realtime services require the 'realtime' extra: "
            "pip install 'vocalbin[realtime]'"
        ) from exc
    return await connect(url, additional_headers=headers)


class OpenAIRealtimeProvider(RealtimeProvider):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        safety_identifier: str | None = None,
        base_url: str = "wss://api.openai.com/v1/realtime",
    ) -> None:
        if safety_identifier is not None and not safety_identifier.strip():
            raise ValueError("safety_identifier must not be blank")
        self._api_key = (
            api_key
            if api_key is not None
            else OpenAICredentials().api_key.get_secret_value()
        )
        self._safety_identifier = safety_identifier
        self._base_url = base_url.rstrip("/")

    def build_url(
        self,
        session_type: Literal["transcription", "translation"],
        model: str,
    ) -> str:
        if session_type == "translation":
            query = urlencode({"model": model})
            return f"{self._base_url}/translations?{query}"
        query = urlencode({"intent": "transcription"})
        return f"{self._base_url}?{query}"

    def build_headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._safety_identifier is not None:
            headers["OpenAI-Safety-Identifier"] = self._safety_identifier
        return headers


class _RealtimeWebSocket:
    def __init__(
        self,
        provider: RealtimeProvider,
        session_type: Literal["transcription", "translation"],
        model: str,
    ) -> None:
        self._provider = provider
        self._session_type = session_type
        self._model = model
        self._connection: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    async def connect(self) -> None:
        if self._connection is not None:
            await self.close()
        self._connection = await _connect_websocket(
            self._provider.build_url(self._session_type, self._model),
            self._provider.build_headers(),
        )

    async def send(self, payload: JsonObject) -> None:
        if self._connection is None:
            raise RuntimeError("Realtime session is not connected.")
        await self._connection.send(json.dumps(payload))

    async def events(self) -> AsyncIterator[JsonObject]:
        if self._connection is None:
            raise RuntimeError("Realtime session is not connected.")
        async for message in self._connection:
            payload = json.loads(message)
            if not isinstance(payload, dict):
                raise ValueError("Realtime server event must be a JSON object.")
            yield payload

    async def close(self) -> None:
        if self._connection is None:
            return
        connection = self._connection
        self._connection = None
        await connection.close()


class _RealtimeStreamer[EventT: (RealtimeTranscriptionEvent, RealtimeTranslationEvent)](
    ABC
):
    def __init__(
        self,
        *,
        audio_input: AudioInput,
        provider: RealtimeProvider,
        session_type: Literal["transcription", "translation"],
        model: str,
    ) -> None:
        self._audio_input = audio_input
        self._connection = _RealtimeWebSocket(provider, session_type, model)
        self._sender_task: asyncio.Task[None] | None = None
        self._sender_error: Exception | None = None
        self._stream_taken = False
        self._session_started = False
        self._stop_called = False
        self._input_finished = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    def stream(self) -> AsyncIterator[EventT]:
        if self._stream_taken:
            raise RuntimeError("Realtime sessions are single-use.")
        if self._stop_called:
            raise RuntimeError("Realtime session has already been stopped.")
        self._stream_taken = True
        return self._stream_events()

    async def stop(self) -> None:
        if self._stop_called:
            return
        self._stop_called = True
        await self._audio_input.stop()
        if self._sender_task is not None and not self._sender_task.done():
            self._sender_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._sender_task
        self._sender_task = None
        await self._connection.close()

    async def _stream_events(self) -> AsyncIterator[EventT]:
        await self._connection.connect()
        await self._connection.send(self._build_session_update())
        self._session_started = True
        await self._audio_input.start()
        self._sender_task = asyncio.create_task(self._send_audio())

        try:
            yield cast(EventT, RealtimeSessionConnected())
            if self._stop_called:
                return
            async for payload in self._connection.events():
                event = self._map_event(payload)
                if event is None:
                    continue
                yield event
                if self._should_finish(event):
                    break
            if self._sender_error is not None:
                raise self._sender_error
        finally:
            await self.stop()

    async def _send_audio(self) -> None:
        try:
            async for chunk in self._audio_input.stream_chunks():
                if self._stop_called or not self._connection.is_connected:
                    return
                if not chunk:
                    continue
                await self._connection.send(self._build_audio_append(chunk))
            self._input_finished = True
            if not self._stop_called and self._connection.is_connected:
                await self._connection.send(self._build_input_finished())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._sender_error = exc
            await self._connection.close()

    @abstractmethod
    def _build_session_update(self) -> JsonObject: ...

    @abstractmethod
    def _build_audio_append(self, chunk: bytes) -> JsonObject: ...

    @abstractmethod
    def _build_input_finished(self) -> JsonObject: ...

    @abstractmethod
    def _map_event(self, payload: JsonObject) -> EventT | None: ...

    @abstractmethod
    def _should_finish(self, event: EventT) -> bool: ...


class OpenAIRealtimeTranscriber(
    _RealtimeStreamer[RealtimeTranscriptionEvent],
    RealtimeTranscription,
):
    def __init__(
        self,
        config: RealtimeTranscriptionConfig | None = None,
        *,
        audio_input: AudioInput | None = None,
        provider: RealtimeProvider | None = None,
        api_key: str | None = None,
        safety_identifier: str | None = None,
    ) -> None:
        if provider is not None and (
            api_key is not None or safety_identifier is not None
        ):
            raise ValueError("Pass either 'provider' or OpenAI credentials, not both.")
        self.config = config or RealtimeTranscriptionConfig()
        selected_provider = provider or OpenAIRealtimeProvider(
            api_key,
            safety_identifier=safety_identifier,
        )
        super().__init__(
            audio_input=audio_input or MicrophoneInput(),
            provider=selected_provider,
            session_type="transcription",
            model=self.config.model,
        )

    async def flush(self) -> None:
        if not self._session_started:
            raise RuntimeError("Cannot flush before stream() has started.")
        if not self._connection.is_connected:
            raise RuntimeError("Cannot flush after the realtime session has closed.")
        await self._connection.send(self._build_input_finished())

    def _build_session_update(self) -> JsonObject:
        transcription: JsonObject = {
            "model": self.config.model,
            "delay": self.config.delay,
        }
        if self.config.language is not None:
            transcription["language"] = self.config.language
        audio_input: JsonObject = {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": transcription,
            "turn_detection": None,
            "noise_reduction": (
                {"type": self.config.noise_reduction}
                if self.config.noise_reduction is not None
                else None
            ),
        }
        session: JsonObject = {
            "type": "transcription",
            "audio": {"input": audio_input},
        }
        if self.config.include_logprobs:
            session["include"] = ["item.input_audio_transcription.logprobs"]
        return {"type": "session.update", "session": session}

    def _build_audio_append(self, chunk: bytes) -> JsonObject:
        return {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }

    def _build_input_finished(self) -> JsonObject:
        return {"type": "input_audio_buffer.commit"}

    def _map_event(self, payload: JsonObject) -> RealtimeTranscriptionEvent | None:
        return _map_transcription_event(payload)

    def _should_finish(self, event: RealtimeTranscriptionEvent) -> bool:
        return self._input_finished and isinstance(event, RealtimeTranscriptCompleted)


class OpenAIRealtimeTranslator(
    _RealtimeStreamer[RealtimeTranslationEvent],
    RealtimeTranslation,
):
    def __init__(
        self,
        config: RealtimeTranslationConfig,
        *,
        audio_input: AudioInput | None = None,
        provider: RealtimeProvider | None = None,
        api_key: str | None = None,
        safety_identifier: str | None = None,
    ) -> None:
        if provider is not None and (
            api_key is not None or safety_identifier is not None
        ):
            raise ValueError("Pass either 'provider' or OpenAI credentials, not both.")
        self.config = config
        selected_provider = provider or OpenAIRealtimeProvider(
            api_key,
            safety_identifier=safety_identifier,
        )
        super().__init__(
            audio_input=audio_input or MicrophoneInput(),
            provider=selected_provider,
            session_type="translation",
            model=self.config.model,
        )

    def _build_session_update(self) -> JsonObject:
        transcription = (
            {"model": "gpt-realtime-whisper"}
            if self.config.include_source_transcript
            else None
        )
        return {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "transcription": transcription,
                        "noise_reduction": (
                            {"type": self.config.noise_reduction}
                            if self.config.noise_reduction is not None
                            else None
                        ),
                    },
                    "output": {"language": self.config.target_language},
                }
            },
        }

    def _build_audio_append(self, chunk: bytes) -> JsonObject:
        return {
            "type": "session.input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }

    def _build_input_finished(self) -> JsonObject:
        return {"type": "session.close"}

    def _map_event(self, payload: JsonObject) -> RealtimeTranslationEvent | None:
        return _map_translation_event(payload)

    def _should_finish(self, event: RealtimeTranslationEvent) -> bool:
        return isinstance(event, RealtimeTranslationClosed)


def _map_error(payload: JsonObject) -> RealtimeError:
    error = payload["error"]
    return RealtimeError(
        error=RealtimeErrorDetails(
            type=error["type"],
            message=error["message"],
            code=error.get("code"),
            event_id=error.get("event_id"),
            param=error.get("param"),
        )
    )


def _map_transcription_event(
    payload: JsonObject,
) -> RealtimeTranscriptionEvent | None:
    match payload.get("type"):
        case "conversation.item.input_audio_transcription.delta":
            return RealtimeTranscriptDelta(
                delta=payload["delta"],
                item_id=payload["item_id"],
                event_id=payload.get("event_id"),
                logprobs=payload.get("logprobs"),
            )
        case "conversation.item.input_audio_transcription.completed":
            return RealtimeTranscriptCompleted(
                transcript=payload["transcript"],
                item_id=payload["item_id"],
                event_id=payload.get("event_id"),
                logprobs=payload.get("logprobs"),
                usage=payload.get("usage"),
            )
        case "input_audio_buffer.speech_started":
            return RealtimeSpeechStarted(
                item_id=payload["item_id"],
                audio_start_ms=payload["audio_start_ms"],
            )
        case "input_audio_buffer.speech_stopped":
            return RealtimeSpeechStopped(
                item_id=payload["item_id"],
                audio_end_ms=payload["audio_end_ms"],
            )
        case "error":
            return _map_error(payload)
        case _:
            return None


def _map_translation_event(payload: JsonObject) -> RealtimeTranslationEvent | None:
    match payload.get("type"):
        case "session.input_transcript.delta":
            return RealtimeSourceTranscriptDelta(
                delta=payload["delta"],
                elapsed_ms=payload.get("elapsed_ms"),
                event_id=payload.get("event_id"),
            )
        case "session.output_transcript.delta":
            return RealtimeTranslationTranscriptDelta(
                delta=payload["delta"],
                elapsed_ms=payload.get("elapsed_ms"),
                event_id=payload.get("event_id"),
            )
        case "session.output_audio.delta":
            return RealtimeTranslationAudioDelta(
                audio=base64.b64decode(payload["delta"]),
                elapsed_ms=payload.get("elapsed_ms"),
                sample_rate=payload.get("sample_rate", 24000),
                channels=payload.get("channels", 1),
                format=payload.get("format", "pcm16"),
                event_id=payload.get("event_id"),
            )
        case "session.closed":
            return RealtimeTranslationClosed(event_id=payload.get("event_id"))
        case "error":
            return _map_error(payload)
        case _:
            return None
