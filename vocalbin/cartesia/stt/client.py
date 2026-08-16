from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, overload

from vocalbin import ports
from vocalbin.cartesia.credentials import Credentials
from vocalbin.cartesia.events import (
    Connected,
    Event,
    TurnEagerEnd,
    TurnEnd,
    TurnResume,
    TurnStart,
    TurnUpdate,
)
from vocalbin.cartesia.stt.models import (
    SpeechToTextConfig,
    SpeechToTextEncoding,
    SpeechToTextModel,
)

if TYPE_CHECKING:
    from cartesia import AsyncCartesia
    from cartesia.resources.stt.auto_finalize import (
        AsyncAutoFinalizeResourceConnection,
    )
    from cartesia.types.stt import STTAutoFinalizeWebsocketResponse


class SpeechToTextError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        doc_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
        self.request_id = request_id
        self.doc_url = doc_url


class SpeechToText(ports.StreamingSpeechToText[SpeechToTextConfig, Event]):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncCartesia | None = None,
        model: SpeechToTextModel | str = SpeechToTextModel.INK_2,
        encoding: SpeechToTextEncoding = SpeechToTextEncoding.PCM_S16LE,
        sample_rate: int = 16000,
        keyterms: list[str] | None = None,
        turn_start_threshold: float | None = None,
        turn_eager_end_threshold: float | None = None,
        turn_end_threshold: float | None = None,
        turn_end_timeout_ms: float | None = None,
    ) -> None:
        if api_key is not None and client is not None:
            raise ValueError("Pass either 'api_key' or 'client', not both.")
        owns_client = client is None
        if client is None:
            resolved_api_key = (
                api_key
                if api_key is not None
                else Credentials().api_key.get_secret_value()
            )
            client = _create_client(resolved_api_key)
        self.client = client
        self.default_config = SpeechToTextConfig(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            keyterms=keyterms,
            turn_start_threshold=turn_start_threshold,
            turn_eager_end_threshold=turn_eager_end_threshold,
            turn_end_threshold=turn_end_threshold,
            turn_end_timeout_ms=turn_end_timeout_ms,
        )
        self._owns_client = owns_client

    @overload
    def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        model: SpeechToTextModel | str = SpeechToTextModel.INK_2,
        encoding: SpeechToTextEncoding = SpeechToTextEncoding.PCM_S16LE,
        sample_rate: int = 16000,
        keyterms: list[str] | None = None,
        turn_start_threshold: float | None = None,
        turn_eager_end_threshold: float | None = None,
        turn_end_threshold: float | None = None,
        turn_end_timeout_ms: float | None = None,
    ) -> AsyncGenerator[Event]: ...

    @overload
    def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        config: SpeechToTextConfig | None = None,
    ) -> AsyncGenerator[Event]: ...

    async def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        model: SpeechToTextModel | str | None = None,
        encoding: SpeechToTextEncoding | None = None,
        sample_rate: int | None = None,
        keyterms: list[str] | None = None,
        turn_start_threshold: float | None = None,
        turn_eager_end_threshold: float | None = None,
        turn_end_threshold: float | None = None,
        turn_end_timeout_ms: float | None = None,
        config: SpeechToTextConfig | None = None,
    ) -> AsyncGenerator[Event]:
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            keyterms=keyterms,
            turn_start_threshold=turn_start_threshold,
            turn_eager_end_threshold=turn_eager_end_threshold,
            turn_end_threshold=turn_end_threshold,
            turn_end_timeout_ms=turn_end_timeout_ms,
        )
        manager = self.client.stt.auto_finalize.websocket(
            **resolved_config.model_dump(
                exclude_none=True,
                mode="json",
                by_alias=True,
            )
        )

        async with manager as connection:
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
                        event = receiver.result()
                    except StopAsyncIteration:
                        break

                    if event.type == "error":
                        raise SpeechToTextError(
                            event.message,
                            error_code=event.error_code,
                            status_code=event.status_code,
                            request_id=event.request_id,
                            doc_url=event.doc_url,
                        )
                    yield _normalize_event(event)
                    receiver = asyncio.ensure_future(anext(responses))
            finally:
                await _stop_task(receiver)
                await _stop_task(sender)

    async def _send_audio(
        self,
        connection: AsyncAutoFinalizeResourceConnection,
        audio: AsyncIterable[bytes],
    ) -> None:
        sent_audio = False
        async for chunk in audio:
            if not chunk:
                continue
            await connection.send_raw(chunk)
            sent_audio = True
        if not sent_audio:
            raise ValueError(
                "The audio stream must contain at least one non-empty chunk."
            )
        await connection.send({"type": "close"})

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _create_client(api_key: str) -> AsyncCartesia:
    try:
        from cartesia import AsyncCartesia
    except ImportError as exc:
        raise ImportError("Cartesia support requires `vocalbin[cartesia]`.") from exc
    return AsyncCartesia(api_key=api_key)


def _resolve_call_config(
    *,
    config: SpeechToTextConfig | None,
    default_config: SpeechToTextConfig,
    model: SpeechToTextModel | str | None,
    encoding: SpeechToTextEncoding | None,
    sample_rate: int | None,
    keyterms: list[str] | None,
    turn_start_threshold: float | None,
    turn_eager_end_threshold: float | None,
    turn_end_threshold: float | None,
    turn_end_timeout_ms: float | None,
) -> SpeechToTextConfig:
    flat_values = (
        model,
        encoding,
        sample_rate,
        keyterms,
        turn_start_threshold,
        turn_eager_end_threshold,
        turn_end_threshold,
        turn_end_timeout_ms,
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
        "turn_start_threshold": turn_start_threshold,
        "turn_eager_end_threshold": turn_eager_end_threshold,
        "turn_end_threshold": turn_end_threshold,
        "turn_end_timeout_ms": turn_end_timeout_ms,
    }
    if model is not None:
        values["model"] = model
    if encoding is not None:
        values["encoding"] = encoding
    if sample_rate is not None:
        values["sample_rate"] = sample_rate
    return SpeechToTextConfig(**values)


async def _stop_task(task: asyncio.Future[Any]) -> None:
    if not task.done():
        task.cancel()
    with suppress(Exception, asyncio.CancelledError):
        await task


def _normalize_event(
    event: STTAutoFinalizeWebsocketResponse,
) -> Event:
    match event.type:
        case "connected":
            return Connected(request_id=event.request_id)
        case "turn.start":
            return TurnStart(request_id=event.request_id)
        case "turn.update":
            return TurnUpdate(
                request_id=event.request_id,
                transcript=event.transcript,
            )
        case "turn.eager_end":
            return TurnEagerEnd(
                request_id=event.request_id,
                transcript=event.transcript,
            )
        case "turn.resume":
            return TurnResume(request_id=event.request_id)
        case "turn.end":
            return TurnEnd(
                request_id=event.request_id,
                transcript=event.transcript,
            )
        case _:
            raise ValueError(f"Unsupported Cartesia STT event type: {event.type}")
