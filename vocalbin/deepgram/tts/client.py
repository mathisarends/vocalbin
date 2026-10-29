from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, overload

from vocalbin import ports
from vocalbin.deepgram.shared import DeepgramClientOwner
from vocalbin.deepgram.tts._connection import _StreamingConnection
from vocalbin.deepgram.tts.models import (
    AudioContainer,
    AudioEncoding,
    TextToSpeechConfig,
    TextToSpeechModel,
    TextToSpeechResponse,
)
from vocalbin.ports import Reset, merge_config

if TYPE_CHECKING:
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
    ports.Connectable,
    ports.TextToSpeech[TextToSpeechConfig, TextToSpeechResponse],
    ports.IncrementalTextToSpeech[TextToSpeechConfig, bytes],
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncDeepgramClient | None = None,
        model: TextToSpeechModel | str = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding = AudioEncoding.LINEAR16,
        container: AudioContainer | None = None,
        sample_rate: int | None = None,
        bit_rate: int | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = TextToSpeechConfig(
            model=model,
            encoding=encoding,
            container=container,
            bit_rate=bit_rate,
            **({"sample_rate": sample_rate} if sample_rate is not None else {}),
        )
        self._streaming = _StreamingConnection(self.client)

    @overload
    async def generate(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | Reset = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding | Reset = AudioEncoding.LINEAR16,
        container: AudioContainer | Reset | None = None,
        sample_rate: int | Reset | None = None,
        bit_rate: int | Reset | None = None,
    ) -> TextToSpeechResponse: ...

    @overload
    async def generate(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> TextToSpeechResponse: ...

    async def generate(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | Reset | None = None,
        encoding: AudioEncoding | Reset | None = None,
        container: AudioContainer | Reset | None = None,
        sample_rate: int | Reset | None = None,
        bit_rate: int | Reset | None = None,
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
            container=resolved_config.effective_container,
            sample_rate=resolved_config.effective_sample_rate,
            content_type=resolved_config.content_type,
        )

    @overload
    def stream(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | Reset = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding | Reset = AudioEncoding.LINEAR16,
        sample_rate: int | Reset | None = None,
    ) -> AsyncGenerator[bytes]: ...

    @overload
    def stream(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> AsyncGenerator[bytes]: ...

    async def stream(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | Reset | None = None,
        encoding: AudioEncoding | Reset | None = None,
        sample_rate: int | Reset | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> AsyncGenerator[bytes]:
        text = _require_valid_text(text)

        async def text_chunks() -> AsyncIterator[str]:
            yield text

        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            encoding=encoding,
            container=None,
            sample_rate=sample_rate,
            bit_rate=None,
        )
        stream = self.stream_incremental(text_chunks(), config=resolved_config)
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
        model: TextToSpeechModel | str | Reset = TextToSpeechModel.AURA_2_THALIA_EN,
        encoding: AudioEncoding | Reset = AudioEncoding.LINEAR16,
        sample_rate: int | Reset | None = None,
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
        model: TextToSpeechModel | str | Reset | None = None,
        encoding: AudioEncoding | Reset | None = None,
        sample_rate: int | Reset | None = None,
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
        _validate_streaming_config(resolved_config)

        async with self._streaming.acquire(_connect_params(resolved_config)) as lease:
            connection = lease.connection
            sender = asyncio.create_task(self._send_text(connection, text))
            responses = connection.__aiter__()
            receiver = asyncio.ensure_future(anext(responses))
            sender_pending = True

            try:
                while True:
                    pending: set[asyncio.Future[Any]] = {receiver}
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
                        lease.complete = True
                        break

                    receiver = asyncio.ensure_future(anext(responses))
            finally:
                await _stop_task(receiver)
                await _stop_task(sender)

    async def connect(self) -> None:
        _validate_streaming_config(self.default_config)
        await self._streaming.connect(_connect_params(self.default_config))

    async def aclose(self) -> None:
        try:
            await self._streaming.aclose()
        finally:
            await super().aclose()

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


def _validate_streaming_config(config: TextToSpeechConfig) -> None:
    if config.encoding not in _STREAMING_ENCODINGS:
        raise ValueError(
            "Deepgram WebSocket streaming requires linear16, mulaw, or alaw."
        )
    if config.container not in (None, AudioContainer.NONE):
        raise ValueError("Deepgram WebSocket streaming requires no container.")


def _text_message(text: str) -> Any:
    from deepgram.extensions.types.sockets import SpeakV1TextMessage

    return SpeakV1TextMessage(type="Speak", text=text)


def _flush_message() -> Any:
    from deepgram.extensions.types.sockets import SpeakV1ControlMessage

    return SpeakV1ControlMessage(type="Flush")


def _connect_params(config: TextToSpeechConfig) -> dict[str, str]:
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
    model: TextToSpeechModel | str | Reset | None,
    encoding: AudioEncoding | Reset | None,
    container: AudioContainer | Reset | None,
    sample_rate: int | Reset | None,
    bit_rate: int | Reset | None,
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
    return merge_config(default_config, TextToSpeechConfig, values)
