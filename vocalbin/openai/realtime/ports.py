from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from vocalbin.openai.realtime.events import TranscriptionEvent, TranslationEvent
from vocalbin.openai.realtime.models import SessionType


class AudioInput(ABC):
    @property
    @abstractmethod
    def is_active(self) -> bool: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def stream_chunks(self) -> AsyncIterator[bytes]: ...


class Provider(ABC):
    @abstractmethod
    def build_url(
        self,
        session_type: SessionType,
        model: str,
    ) -> str: ...

    @abstractmethod
    def build_headers(self) -> dict[str, str]: ...


class WebSocketSession(ABC):
    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...


class Transcription(WebSocketSession):
    @abstractmethod
    def stream(self) -> AsyncIterator[TranscriptionEvent]: ...

    @abstractmethod
    async def flush(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class Translation(WebSocketSession):
    @abstractmethod
    def stream(self) -> AsyncIterator[TranslationEvent]: ...

    @abstractmethod
    async def stop(self) -> None: ...
