import asyncio
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
    RealtimeSessionConnected,
    RealtimeSessionType,
    RealtimeTranscriptCompleted,
    RealtimeTranscriptionConfig,
    RealtimeTranscriptionEvent,
    RealtimeTranslationClosed,
    RealtimeTranslationConfig,
    RealtimeTranslationEvent,
)
from vocalbin.ports import (
    AudioInput,
    RealtimeProvider,
    RealtimeTranscription,
    RealtimeTranslation,
)
from vocalbin.realtime._audio import MicrophoneInput
from vocalbin.realtime._models import (
    TRANSCRIPTION_EVENT_TYPES,
    TRANSLATION_EVENT_TYPES,
    RealtimeAudioAppend,
    RealtimeClientMessage,
    RealtimeInputFinished,
    RealtimeNoiseReductionConfig,
    RealtimePcmFormat,
    RealtimeSessionUpdate,
    RealtimeTranscriptionAudio,
    RealtimeTranscriptionAudioAppend,
    RealtimeTranscriptionAudioCommit,
    RealtimeTranscriptionAudioInput,
    RealtimeTranscriptionSession,
    RealtimeTranscriptionSessionUpdate,
    RealtimeTranscriptionSettings,
    RealtimeTranslationAudio,
    RealtimeTranslationAudioAppend,
    RealtimeTranslationInputAudio,
    RealtimeTranslationOutputAudio,
    RealtimeTranslationSession,
    RealtimeTranslationSessionClose,
    RealtimeTranslationSessionUpdate,
    RealtimeTranslationTranscriptionSettings,
    transcription_event_adapter,
    translation_event_adapter,
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
        session_type: RealtimeSessionType,
        model: str,
    ) -> str:
        if session_type == RealtimeSessionType.TRANSLATION:
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
        session_type: RealtimeSessionType,
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

    async def send(self, payload: RealtimeClientMessage | dict[str, Any]) -> None:
        if self._connection is None:
            raise RuntimeError("Realtime session is not connected.")
        message = (
            payload if isinstance(payload, dict) else payload.model_dump(mode="json")
        )
        await self._connection.send(json.dumps(message))

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


class _RealtimeStreamer[EventT: (RealtimeTranscriptionEvent, RealtimeTranslationEvent)](
    ABC
):
    def __init__(
        self,
        *,
        audio_input: AudioInput,
        provider: RealtimeProvider,
        session_type: RealtimeSessionType,
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
    def _build_session_update(self) -> RealtimeSessionUpdate: ...

    @abstractmethod
    def _build_audio_append(self, chunk: bytes) -> RealtimeAudioAppend: ...

    @abstractmethod
    def _build_input_finished(self) -> RealtimeInputFinished: ...

    @abstractmethod
    def _map_event(self, payload: dict[str, Any]) -> EventT | None: ...

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
            session_type=RealtimeSessionType.TRANSCRIPTION,
            model=self.config.model,
        )

    async def flush(self) -> None:
        if not self._session_started:
            raise RuntimeError("Cannot flush before stream() has started.")
        if not self._connection.is_connected:
            raise RuntimeError("Cannot flush after the realtime session has closed.")
        await self._connection.send(self._build_input_finished())

    def _build_session_update(self) -> RealtimeTranscriptionSessionUpdate:
        noise_reduction = (
            RealtimeNoiseReductionConfig(type=self.config.noise_reduction)
            if self.config.noise_reduction is not None
            else None
        )
        include: list[Literal["item.input_audio_transcription.logprobs"]] | None = (
            ["item.input_audio_transcription.logprobs"]
            if self.config.include_logprobs
            else None
        )
        return RealtimeTranscriptionSessionUpdate(
            session=RealtimeTranscriptionSession(
                audio=RealtimeTranscriptionAudio(
                    input=RealtimeTranscriptionAudioInput(
                        format=RealtimePcmFormat(),
                        transcription=RealtimeTranscriptionSettings(
                            model=self.config.model,
                            delay=self.config.delay,
                            language=self.config.language,
                        ),
                        noise_reduction=noise_reduction,
                    )
                ),
                include=include,
            ),
        )

    def _build_audio_append(self, chunk: bytes) -> RealtimeTranscriptionAudioAppend:
        return RealtimeTranscriptionAudioAppend.from_audio(chunk)

    def _build_input_finished(self) -> RealtimeTranscriptionAudioCommit:
        return RealtimeTranscriptionAudioCommit()

    def _map_event(self, payload: dict[str, Any]) -> RealtimeTranscriptionEvent | None:
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
            session_type=RealtimeSessionType.TRANSLATION,
            model=self.config.model,
        )

    def _build_session_update(self) -> RealtimeTranslationSessionUpdate:
        transcription = (
            RealtimeTranslationTranscriptionSettings()
            if self.config.include_source_transcript
            else None
        )
        noise_reduction = (
            RealtimeNoiseReductionConfig(type=self.config.noise_reduction)
            if self.config.noise_reduction is not None
            else None
        )
        return RealtimeTranslationSessionUpdate(
            session=RealtimeTranslationSession(
                audio=RealtimeTranslationAudio(
                    input=RealtimeTranslationInputAudio(
                        transcription=transcription,
                        noise_reduction=noise_reduction,
                    ),
                    output=RealtimeTranslationOutputAudio(
                        language=self.config.target_language
                    ),
                )
            )
        )

    def _build_audio_append(self, chunk: bytes) -> RealtimeTranslationAudioAppend:
        return RealtimeTranslationAudioAppend.from_audio(chunk)

    def _build_input_finished(self) -> RealtimeTranslationSessionClose:
        return RealtimeTranslationSessionClose()

    def _map_event(self, payload: dict[str, Any]) -> RealtimeTranslationEvent | None:
        return _map_translation_event(payload)

    def _should_finish(self, event: RealtimeTranslationEvent) -> bool:
        return isinstance(event, RealtimeTranslationClosed)


def _map_transcription_event(
    payload: dict[str, Any],
) -> RealtimeTranscriptionEvent | None:
    if payload.get("type") not in TRANSCRIPTION_EVENT_TYPES:
        return None
    return transcription_event_adapter.validate_python(payload).to_event()


def _map_translation_event(
    payload: dict[str, Any],
) -> RealtimeTranslationEvent | None:
    if payload.get("type") not in TRANSLATION_EVENT_TYPES:
        return None
    return translation_event_adapter.validate_python(payload).to_event()
