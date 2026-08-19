from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast, overload

from vocalbin import ports
from vocalbin.deepgram.shared import DeepgramClientOwner
from vocalbin.deepgram.tts.models import (
    AudioContainer,
    AudioEncoding,
    TextToSpeechConfig,
    TextToSpeechModel,
    TextToSpeechResponse,
)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from deepgram import AsyncDeepgramClient
    from deepgram.speak.v1.socket_client import AsyncV1SocketClient

_MAX_TEXT_LENGTH = 2000
_STREAMING_ENCODINGS = {
    AudioEncoding.LINEAR16,
    AudioEncoding.MULAW,
    AudioEncoding.ALAW,
}


class TextToSpeechError(RuntimeError):
    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class TextToSpeech(
    DeepgramClientOwner,
    ports.TextToSpeech[TextToSpeechConfig, TextToSpeechResponse],
    ports.StreamingTextToSpeech[TextToSpeechConfig, bytes],
    ports.WebSocketClient,
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncDeepgramClient | None = None,
        model: TextToSpeechModel | str = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding = AudioEncoding.LINEAR16,
        container: AudioContainer | None = None,
        sample_rate: int | None = 24000,
        bit_rate: int | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = TextToSpeechConfig(
            model=model,
            encoding=encoding,
            container=container,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
        )
        self._connection: AsyncV1SocketClient | None = None
        self._connection_manager: (
            AbstractAsyncContextManager[AsyncV1SocketClient] | None
        ) = None
        self._connection_config: TextToSpeechConfig | None = None
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

    async def aclose(self) -> None:
        await self.disconnect()
        await super().aclose()

    @overload
    async def generate(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding = AudioEncoding.LINEAR16,
        container: AudioContainer | None = None,
        sample_rate: int | None = None,
        bit_rate: int | None = None,
    ) -> TextToSpeechResponse: ...

    @overload
    async def generate(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> TextToSpeechResponse: ...

    async def generate(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | None = None,
        encoding: AudioEncoding | None = None,
        container: AudioContainer | None = None,
        sample_rate: int | None = None,
        bit_rate: int | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> TextToSpeechResponse:
        text = _require_valid_text(text)
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            encoding=encoding,
            container=container,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
        )
        params = resolved_config.model_dump(exclude_none=True, mode="json")
        audio = bytearray()
        async for chunk in self.client.speak.v1.audio.generate(text=text, **params):
            audio.extend(chunk)

        return TextToSpeechResponse(
            audio=bytes(audio),
            model=resolved_config.model,
            encoding=resolved_config.encoding,
            container=resolved_config.container,
            sample_rate=resolved_config.sample_rate,
            content_type=resolved_config.content_type,
        )

    @overload
    def stream(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding = AudioEncoding.LINEAR16,
        sample_rate: int | None = None,
    ) -> AsyncGenerator[bytes]: ...

    @overload
    def stream(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> AsyncGenerator[bytes]: ...

    async def stream(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | None = None,
        encoding: AudioEncoding | None = None,
        sample_rate: int | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> AsyncGenerator[bytes]:
        text = _require_valid_text(text)

        async def text_chunks() -> AsyncIterator[str]:
            yield text

        stream = self.stream_incremental(
            text_chunks(),
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            config=config,
        )
        try:
            async for audio in stream:
                yield audio
        finally:
            await stream.aclose()

    @overload
    def stream_incremental(
        self,
        text: AsyncIterable[str],
        *,
        model: TextToSpeechModel | str = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding = AudioEncoding.LINEAR16,
        sample_rate: int | None = None,
    ) -> AsyncGenerator[bytes]: ...

    @overload
    def stream_incremental(
        self,
        text: AsyncIterable[str],
        *,
        config: TextToSpeechConfig | None = None,
    ) -> AsyncGenerator[bytes]: ...

    async def stream_incremental(
        self,
        text: AsyncIterable[str],
        *,
        model: TextToSpeechModel | str | None = None,
        encoding: AudioEncoding | None = None,
        sample_rate: int | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> AsyncGenerator[bytes]:
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            encoding=encoding,
            container=None,
            sample_rate=sample_rate,
            bit_rate=None,
        )
        if resolved_config.encoding not in _STREAMING_ENCODINGS:
            raise ValueError(
                "Deepgram WebSocket streaming requires linear16, mulaw, or alaw."
            )
        if resolved_config.container not in (None, AudioContainer.NONE):
            raise ValueError("Deepgram WebSocket streaming requires no container.")

        connection = await self._get_connection(resolved_config)
        sender = asyncio.create_task(self._send_text(connection, text))
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
                    message = receiver.result()
                except StopAsyncIteration:
                    break

                if isinstance(message, bytes):
                    yield message
                elif message.type == "Warning":
                    raise TextToSpeechError(
                        message.warn_msg, error_code=message.warn_code
                    )
                elif message.type == "Flushed":
                    break

                receiver = asyncio.ensure_future(anext(responses))
        finally:
            await _stop_task(receiver)
            await _stop_task(sender)
            await self.disconnect()

    async def _send_text(
        self, connection: AsyncV1SocketClient, text: AsyncIterable[str]
    ) -> None:
        sent_text = False
        async for chunk in text:
            if not chunk:
                continue
            await connection.send_text(_text_message(chunk))
            sent_text = True
        if not sent_text:
            raise ValueError(
                "The text stream must contain at least one non-empty chunk."
            )
        await connection.send_control(_flush_message())

    async def _get_connection(self, config: TextToSpeechConfig) -> AsyncV1SocketClient:
        async with self._connection_lock:
            if self._connection is not None:
                if self._connection_config == config:
                    return self._connection
                manager = cast(Any, self._connection_manager)
                self._connection = None
                self._connection_manager = None
                self._connection_config = None
                await manager.__aexit__(None, None, None)
            manager = self.client.speak.v1.connect(**_connect_params(config))
            self._connection = await manager.__aenter__()
            self._connection_manager = manager
            self._connection_config = config
            return self._connection


def _text_message(text: str) -> Any:
    from deepgram.extensions.types.sockets import SpeakV1TextMessage

    return SpeakV1TextMessage(type="Speak", text=text)


def _flush_message() -> Any:
    from deepgram.extensions.types.sockets import SpeakV1ControlMessage

    return SpeakV1ControlMessage(type="Flush")


def _connect_params(config: TextToSpeechConfig) -> dict[str, Any]:
    # Deepgram sends every websocket option as a query string parameter.
    params = config.model_dump(
        include={"model", "encoding", "sample_rate"}, exclude_none=True, mode="json"
    )
    return {key: str(value) for key, value in params.items()}


async def _stop_task(task: asyncio.Future[Any]) -> None:
    if not task.done():
        task.cancel()
    with suppress(Exception, asyncio.CancelledError):
        await task


def _require_valid_text(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be blank")
    if len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(f"text must not exceed {_MAX_TEXT_LENGTH} characters")
    return text


def _resolve_call_config(
    *,
    config: TextToSpeechConfig | None,
    default_config: TextToSpeechConfig,
    model: TextToSpeechModel | str | None,
    encoding: AudioEncoding | None,
    container: AudioContainer | None,
    sample_rate: int | None,
    bit_rate: int | None,
) -> TextToSpeechConfig:
    flat_values = (model, encoding, container, sample_rate, bit_rate)
    has_flat_values = any(value is not None for value in flat_values)
    if config is not None:
        if has_flat_values:
            raise ValueError("Pass either 'config' or flat parameters, not both.")
        return config
    if not has_flat_values:
        return default_config

    values: dict[str, Any] = {"container": container, "bit_rate": bit_rate}
    if model is not None:
        values["model"] = model
    if encoding is not None:
        values["encoding"] = encoding
    if sample_rate is not None:
        values["sample_rate"] = sample_rate
    return TextToSpeechConfig(**values)
