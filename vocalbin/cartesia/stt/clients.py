from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from vocalbin.cartesia.credentials import CartesiaCredentials
from vocalbin.cartesia.stt.models import (
    CartesiaSpeechToTextConfig,
    CartesiaSpeechToTextConnected,
    CartesiaSpeechToTextEvent,
    CartesiaSpeechToTextTurnEagerEnd,
    CartesiaSpeechToTextTurnEnd,
    CartesiaSpeechToTextTurnResume,
    CartesiaSpeechToTextTurnStart,
    CartesiaSpeechToTextTurnUpdate,
)
from vocalbin.ports import StreamingSpeechToText

if TYPE_CHECKING:
    from cartesia import AsyncCartesia
    from cartesia.resources.stt.auto_finalize import (
        AsyncAutoFinalizeResourceConnection,
    )
    from cartesia.types.stt import STTAutoFinalizeWebsocketResponse


class CartesiaSpeechToTextError(RuntimeError):
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


class CartesiaSpeechToText(
    StreamingSpeechToText[CartesiaSpeechToTextConfig, CartesiaSpeechToTextEvent]
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncCartesia | None = None,
        default_config: CartesiaSpeechToTextConfig | None = None,
    ) -> None:
        if api_key is not None and client is not None:
            raise ValueError("Pass either 'api_key' or 'client', not both.")
        owns_client = client is None
        if client is None:
            resolved_api_key = (
                api_key
                if api_key is not None
                else CartesiaCredentials().api_key.get_secret_value()
            )
            client = _create_client(resolved_api_key)
        self.client = client
        self.default_config = default_config or CartesiaSpeechToTextConfig()
        self._owns_client = owns_client

    async def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        config: CartesiaSpeechToTextConfig | None = None,
    ) -> AsyncGenerator[CartesiaSpeechToTextEvent]:
        resolved_config = config or self.default_config
        manager = self.client.stt.auto_finalize.websocket(
            **resolved_config.to_cartesia_params()
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
                        raise CartesiaSpeechToTextError(
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


async def _stop_task(task: asyncio.Future[Any]) -> None:
    if not task.done():
        task.cancel()
    with suppress(Exception, asyncio.CancelledError):
        await task


def _normalize_event(
    event: STTAutoFinalizeWebsocketResponse,
) -> CartesiaSpeechToTextEvent:
    match event.type:
        case "connected":
            return CartesiaSpeechToTextConnected(request_id=event.request_id)
        case "turn.start":
            return CartesiaSpeechToTextTurnStart(request_id=event.request_id)
        case "turn.update":
            return CartesiaSpeechToTextTurnUpdate(
                request_id=event.request_id,
                transcript=event.transcript,
            )
        case "turn.eager_end":
            return CartesiaSpeechToTextTurnEagerEnd(
                request_id=event.request_id,
                transcript=event.transcript,
            )
        case "turn.resume":
            return CartesiaSpeechToTextTurnResume(request_id=event.request_id)
        case "turn.end":
            return CartesiaSpeechToTextTurnEnd(
                request_id=event.request_id,
                transcript=event.transcript,
            )
        case _:
            raise ValueError(f"Unsupported Cartesia STT event type: {event.type}")
