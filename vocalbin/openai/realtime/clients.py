import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from types import TracebackType
from typing import Any, Self, cast
from urllib.parse import urlencode

from vocalbin.openai.credentials import Credentials
from vocalbin.openai.realtime import ports
from vocalbin.openai.realtime.audio import MicrophoneInput
from vocalbin.openai.realtime.base import (
    TRANSCRIPTION_SPEC,
    TRANSLATION_SPEC,
    RealtimeSessionSpec,
)
from vocalbin.openai.realtime.events import (
    SessionConnected,
    TranscriptionEvent,
    TranslationEvent,
)
from vocalbin.openai.realtime.messages import (
    RealtimeClientMessage,
    RealtimeSessionUpdate,
    RealtimeTranscriptionSessionUpdate,
    RealtimeTranslationSessionUpdate,
)
from vocalbin.openai.realtime.models import (
    SessionType,
    TranscriptionConfig,
    TranslationConfig,
)


async def _connect_websocket(url: str, headers: dict[str, str]) -> Any:
    try:
        connect = importlib.import_module("websockets.asyncio.client").connect
    except ImportError as exc:
        raise ImportError(
            "Realtime services require the 'realtime' extra: "
            "pip install 'vocalbin[realtime]'"
        ) from exc
    return await connect(url, additional_headers=headers)


class Provider(ports.Provider):
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
            api_key if api_key is not None else Credentials().api_key.get_secret_value()
        )
        self._safety_identifier = safety_identifier
        self._base_url = base_url.rstrip("/")

    def build_url(
        self,
        session_type: SessionType,
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


def _resolve_provider(
    provider: ports.Provider | None,
    api_key: str | None,
    safety_identifier: str | None,
) -> ports.Provider:
    if provider is None:
        return Provider(api_key, safety_identifier=safety_identifier)
    if api_key is not None or safety_identifier is not None:
        raise ValueError("Pass either 'provider' or OpenAI credentials, not both.")
    return provider


class _RealtimeWebSocket:
    def __init__(
        self,
        provider: ports.Provider,
        session_type: SessionType,
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

    async def send(self, message: RealtimeClientMessage) -> None:
        if self._connection is None:
            raise RuntimeError("Realtime session is not connected.")
        await self._connection.send(json.dumps(message.model_dump(mode="json")))

    async def events(self) -> AsyncIterator[dict[str, Any]]:
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


class _RealtimeStreamer[EventT: (TranscriptionEvent, TranslationEvent)]:
    def __init__(
        self,
        *,
        spec: RealtimeSessionSpec[EventT],
        session_update: RealtimeSessionUpdate,
        audio_input: ports.AudioInput,
        provider: ports.Provider,
        model: str,
    ) -> None:
        self._spec = spec
        self._session_update = session_update
        self._audio_input = audio_input
        self._connection = _RealtimeWebSocket(provider, spec.session_type, model)
        self._sender_task: asyncio.Task[None] | None = None
        self._sender_error: Exception | None = None
        self._stream_taken = False
        self._session_started = False
        self._stop_called = False
        self._input_finished = False

    @property
    def is_connected(self) -> bool:
        return self._connection.is_connected

    async def connect(self) -> None:
        if self._stop_called:
            raise RuntimeError("Realtime session has already been stopped.")
        if not self._connection.is_connected:
            await self._connection.connect()

    async def disconnect(self) -> None:
        await self._connection.close()

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
        await self.disconnect()

    async def _stream_events(self) -> AsyncIterator[EventT]:
        await self.connect()
        await self._connection.send(self._session_update)
        self._session_started = True
        await self._audio_input.start()
        self._sender_task = asyncio.create_task(self._send_audio())

        try:
            yield cast(EventT, SessionConnected())
            if self._stop_called:
                return
            async for payload in self._connection.events():
                event = self._spec.parse_event(payload)
                if event is None:
                    continue
                yield event
                if self._spec.is_final_event(
                    event, input_finished=self._input_finished
                ):
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
                await self._connection.send(self._spec.audio_append.from_audio(chunk))
            self._input_finished = True
            if not self._stop_called and self._connection.is_connected:
                await self._connection.send(self._spec.input_finished())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._sender_error = exc
            await self._connection.close()


class Transcriber(
    _RealtimeStreamer[TranscriptionEvent],
    ports.Transcription,
):
    def __init__(
        self,
        config: TranscriptionConfig | None = None,
        *,
        audio_input: ports.AudioInput | None = None,
        provider: ports.Provider | None = None,
        api_key: str | None = None,
        safety_identifier: str | None = None,
    ) -> None:
        resolved_provider = _resolve_provider(provider, api_key, safety_identifier)
        self.config = config or TranscriptionConfig()
        super().__init__(
            spec=TRANSCRIPTION_SPEC,
            session_update=RealtimeTranscriptionSessionUpdate.from_config(self.config),
            audio_input=audio_input or MicrophoneInput(),
            provider=resolved_provider,
            model=self.config.model,
        )

    async def flush(self) -> None:
        if not self._session_started:
            raise RuntimeError("Cannot flush before stream() has started.")
        if not self._connection.is_connected:
            raise RuntimeError("Cannot flush after the realtime session has closed.")
        await self._connection.send(self._spec.input_finished())


class Translator(
    _RealtimeStreamer[TranslationEvent],
    ports.Translation,
):
    def __init__(
        self,
        config: TranslationConfig,
        *,
        audio_input: ports.AudioInput | None = None,
        provider: ports.Provider | None = None,
        api_key: str | None = None,
        safety_identifier: str | None = None,
    ) -> None:
        resolved_provider = _resolve_provider(provider, api_key, safety_identifier)
        self.config = config
        super().__init__(
            spec=TRANSLATION_SPEC,
            session_update=RealtimeTranslationSessionUpdate.from_config(self.config),
            audio_input=audio_input or MicrophoneInput(),
            provider=resolved_provider,
            model=self.config.model,
        )
