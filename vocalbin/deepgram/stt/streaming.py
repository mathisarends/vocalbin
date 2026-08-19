from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast, overload

from vocalbin import ports
from vocalbin.deepgram.events import (
    Connected,
    Event,
    TurnEagerEnd,
    TurnEnd,
    TurnEvent,
    TurnResume,
    TurnStart,
    TurnUpdate,
)
from vocalbin.deepgram.shared import DeepgramClientOwner
from vocalbin.deepgram.stt.models import (
    StreamingSpeechToTextConfig,
    StreamingSpeechToTextEncoding,
    StreamingSpeechToTextModel,
)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from deepgram import AsyncDeepgramClient
    from deepgram.listen.v2.socket_client import (
        AsyncV2SocketClient,
        V2SocketClientResponse,
    )

_TURN_EVENTS: dict[str, type[TurnEvent]] = {
    "StartOfTurn": TurnStart,
    "Update": TurnUpdate,
    "EagerEndOfTurn": TurnEagerEnd,
    "TurnResumed": TurnResume,
    "EndOfTurn": TurnEnd,
}


class SpeechToTextError(RuntimeError):
    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class StreamingSpeechToText(
    DeepgramClientOwner,
    ports.StreamingSpeechToText[StreamingSpeechToTextConfig, Event],
    ports.WebSocketClient,
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncDeepgramClient | None = None,
        model: StreamingSpeechToTextModel
        | str = StreamingSpeechToTextModel.FLUX_GENERAL_EN,
        encoding: StreamingSpeechToTextEncoding = (
            StreamingSpeechToTextEncoding.LINEAR16
        ),
        sample_rate: int = 16000,
        keyterms: list[str] | None = None,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = StreamingSpeechToTextConfig(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            keyterms=keyterms,
            eot_threshold=eot_threshold,
            eager_eot_threshold=eager_eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
        )
        self._connection: AsyncV2SocketClient | None = None
        self._connection_manager: (
            AbstractAsyncContextManager[AsyncV2SocketClient] | None
        ) = None
        self._connection_config: StreamingSpeechToTextConfig | None = None
        self._connection_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    async def connect(self) -> None:
        await self._get_connection(self.default_config)

    async def disconnect(self) -> None:
        async with self._connection_lock:
            manager = self._connection_manager
            self._connection = None
            self._connection_manager = None
            self._connection_config = None
            if manager is not None:
                await manager.__aexit__(None, None, None)

    @overload
    def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        model: StreamingSpeechToTextModel
        | str = StreamingSpeechToTextModel.FLUX_GENERAL_EN,
        encoding: StreamingSpeechToTextEncoding = (
            StreamingSpeechToTextEncoding.LINEAR16
        ),
        sample_rate: int = 16000,
        keyterms: list[str] | None = None,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
    ) -> AsyncGenerator[Event]: ...

    @overload
    def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        config: StreamingSpeechToTextConfig | None = None,
    ) -> AsyncGenerator[Event]: ...

    async def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        model: StreamingSpeechToTextModel | str | None = None,
        encoding: StreamingSpeechToTextEncoding | None = None,
        sample_rate: int | None = None,
        keyterms: list[str] | None = None,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        config: StreamingSpeechToTextConfig | None = None,
    ) -> AsyncGenerator[Event]:
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            keyterms=keyterms,
            eot_threshold=eot_threshold,
            eager_eot_threshold=eager_eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
        )
        connection = await self._get_connection(resolved_config)
        try:
            sender = asyncio.create_task(self._send_audio(connection, audio))
            responses = connection.__aiter__()
            receiver = asyncio.ensure_future(anext(responses))
            sender_pending = True

            try:
                while True:
                    pending = {receiver}
                    if sender_pending:
                        pending.add(sender)
                    done, _ = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )

                    if sender in done:
                        sender_pending = False
                        sender.result()

                    if receiver not in done:
                        continue
                    try:
                        message: V2SocketClientResponse = receiver.result()
                    except StopAsyncIteration:
                        break

                    yield _normalize_event(message)
                    receiver = asyncio.ensure_future(anext(responses))
            finally:
                await _stop_task(receiver)
                await _stop_task(sender)
        finally:
            await self.disconnect()

    async def _get_connection(
        self, config: StreamingSpeechToTextConfig
    ) -> AsyncV2SocketClient:
        async with self._connection_lock:
            if self._connection is not None:
                if self._connection_config == config:
                    return self._connection
                manager = cast(Any, self._connection_manager)
                self._connection = None
                self._connection_manager = None
                self._connection_config = None
                await manager.__aexit__(None, None, None)
            manager = self.client.listen.v2.connect(**_connect_params(config))
            self._connection = await manager.__aenter__()
            self._connection_manager = manager
            self._connection_config = config
            return self._connection

    async def _send_audio(
        self, connection: AsyncV2SocketClient, audio: AsyncIterable[bytes]
    ) -> None:
        sent_audio = False
        async for chunk in audio:
            if not chunk:
                continue
            await connection.send_media(chunk)
            sent_audio = True
        if not sent_audio:
            raise ValueError(
                "The audio stream must contain at least one non-empty chunk."
            )
        await connection.send_control(_close_stream_message())


def _close_stream_message() -> Any:
    from deepgram.extensions.types.sockets import ListenV2ControlMessage

    return ListenV2ControlMessage(type="CloseStream")


def _connect_params(config: StreamingSpeechToTextConfig) -> dict[str, Any]:
    # Deepgram sends every websocket option as a query string parameter.
    params = config.model_dump(exclude_none=True, mode="json", by_alias=True)
    return {
        key: value if isinstance(value, list) else str(value)
        for key, value in params.items()
    }


async def _stop_task(task: asyncio.Future[Any]) -> None:
    if not task.done():
        task.cancel()
    with suppress(Exception, asyncio.CancelledError):
        await task


def _normalize_event(message: V2SocketClientResponse) -> Event:
    # The SDK models ignore `exclude`, so drop the wire discriminator by hand.
    payload = message.model_dump()
    payload.pop("type")
    match message.type:
        case "Connected":
            return Connected.model_validate(payload)
        case "FatalError":
            raise SpeechToTextError(payload["description"], error_code=payload["code"])
        case "TurnInfo":
            event_name = payload.pop("event")
            turn_event = _TURN_EVENTS.get(event_name)
            if turn_event is None:
                raise ValueError(f"Unsupported Deepgram STT turn event: {event_name}")
            return turn_event.model_validate(payload)
        case _:
            raise ValueError(f"Unsupported Deepgram STT message type: {message.type}")


def _resolve_call_config(
    *,
    config: StreamingSpeechToTextConfig | None,
    default_config: StreamingSpeechToTextConfig,
    model: StreamingSpeechToTextModel | str | None,
    encoding: StreamingSpeechToTextEncoding | None,
    sample_rate: int | None,
    keyterms: list[str] | None,
    eot_threshold: float | None,
    eager_eot_threshold: float | None,
    eot_timeout_ms: int | None,
) -> StreamingSpeechToTextConfig:
    flat_values = (
        model,
        encoding,
        sample_rate,
        keyterms,
        eot_threshold,
        eager_eot_threshold,
        eot_timeout_ms,
    )
    has_flat_values = any(value is not None for value in flat_values)
    if config is not None:
        if has_flat_values:
            raise ValueError("Pass either 'config' or flat parameters, not both.")
        return config
    if not has_flat_values:
        return default_config

    values: dict[str, Any] = {
        "keyterms": keyterms,
        "eot_threshold": eot_threshold,
        "eager_eot_threshold": eager_eot_threshold,
        "eot_timeout_ms": eot_timeout_ms,
    }
    if model is not None:
        values["model"] = model
    if encoding is not None:
        values["encoding"] = encoding
    if sample_rate is not None:
        values["sample_rate"] = sample_rate
    return StreamingSpeechToTextConfig(**values)
