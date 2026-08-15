from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from vocalbin.cartesia.credentials import CartesiaCredentials
from vocalbin.cartesia.models import (
    CartesiaTextToSpeechConfig,
    CartesiaTextToSpeechRequest,
    CartesiaTextToSpeechResponse,
)
from vocalbin.ports import StreamingTextToSpeech, TextToSpeech

if TYPE_CHECKING:
    from cartesia import AsyncCartesia
    from cartesia.resources.tts import (
        AsyncTTSResourceConnection,
        AsyncTTSResourceConnectionManager,
        AsyncWebSocketContext,
    )


class CartesiaTextToSpeechError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class CartesiaTextToSpeech(
    TextToSpeech[CartesiaTextToSpeechRequest, CartesiaTextToSpeechResponse],
    StreamingTextToSpeech[CartesiaTextToSpeechRequest, bytes],
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncCartesia | None = None,
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
        self._owns_client = owns_client
        self._connection: AsyncTTSResourceConnection | None = None
        self._connection_manager: AsyncTTSResourceConnectionManager | None = None
        self._connection_lock = asyncio.Lock()

    async def generate(
        self, request: CartesiaTextToSpeechRequest
    ) -> CartesiaTextToSpeechResponse:
        params = request.to_cartesia_params()
        params["transcript"] = request.text
        result = await self.client.tts.generate(**params)

        return CartesiaTextToSpeechResponse(
            audio=await result.read(),
            model=request.model,
            voice_id=request.voice_id,
            output_format=request.output_format,
            content_type=_content_type(request.output_format.container),
        )

    async def stream(
        self, request: CartesiaTextToSpeechRequest
    ) -> AsyncGenerator[bytes]:
        async def text_chunks() -> AsyncIterator[str]:
            yield request.text

        stream = self.stream_text(text_chunks(), request)
        try:
            async for audio in stream:
                yield audio
        finally:
            await stream.aclose()

    async def stream_text(
        self,
        text: AsyncIterable[str],
        config: CartesiaTextToSpeechConfig,
    ) -> AsyncGenerator[bytes]:
        if config.output_format.container != "raw":
            raise ValueError("Cartesia WebSocket streaming requires raw output.")

        connection = await self._get_connection()
        params = config.to_cartesia_params()
        context = connection.context(
            timeout=config.timeout,
            max_buffer_delay_ms=config.max_buffer_delay_ms,
            **params,
        )
        sender = asyncio.create_task(self._send_text(context, text))
        responses = context.receive().__aiter__()
        receiver = asyncio.ensure_future(anext(responses))
        sender_pending = True
        terminal_event = False

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

                if event.type == "chunk" and event.audio:
                    yield event.audio
                elif event.type == "error":
                    terminal_event = True
                    message = event.message or event.title or "Cartesia TTS failed"
                    raise CartesiaTextToSpeechError(
                        message,
                        error_code=event.error_code,
                        status_code=event.status_code,
                    )
                elif event.type == "done":
                    terminal_event = True
                    break

                receiver = asyncio.ensure_future(anext(responses))
        finally:
            await _stop_task(receiver)
            await _stop_task(sender)
            if not terminal_event:
                await context.cancel()

    async def _send_text(
        self, context: AsyncWebSocketContext, text: AsyncIterable[str]
    ) -> None:
        sent_text = False
        async for chunk in text:
            if not chunk:
                continue
            await context.push(chunk)
            sent_text = True
        if not sent_text:
            raise ValueError(
                "The text stream must contain at least one non-empty chunk."
            )
        await context.no_more_inputs()

    async def _get_connection(self) -> AsyncTTSResourceConnection:
        if self._connection is not None:
            return self._connection
        async with self._connection_lock:
            if self._connection is None:
                manager = self.client.tts.websocket_connect()
                self._connection = await manager.__aenter__()
                self._connection_manager = manager
        return self._connection

    async def aclose(self) -> None:
        if self._connection_manager is not None:
            await self._connection_manager.__aexit__(None, None, None)
            self._connection = None
            self._connection_manager = None
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
    client = AsyncCartesia(api_key=api_key)
    return client


async def _stop_task(task: asyncio.Future[Any]) -> None:
    if not task.done():
        task.cancel()
    with suppress(Exception, asyncio.CancelledError):
        await task


def _content_type(container: str) -> str:
    return {
        "raw": "audio/pcm",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
    }[container]
