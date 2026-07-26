from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from vocalbin.realtime.models import (
    RealtimeSessionType,
    RealtimeTranscriptionEvent,
    RealtimeTranslationEvent,
)


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


class RealtimeProvider(ABC):
    @abstractmethod
    def build_url(
        self,
        session_type: RealtimeSessionType,
        model: str,
    ) -> str: ...

    @abstractmethod
    def build_headers(self) -> dict[str, str]: ...


class RealtimeTranscription(ABC):
    @abstractmethod
    def stream(self) -> AsyncIterator[RealtimeTranscriptionEvent]: ...

    @abstractmethod
    async def flush(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class RealtimeTranslation(ABC):
    @abstractmethod
    def stream(self) -> AsyncIterator[RealtimeTranslationEvent]: ...

    @abstractmethod
    async def stop(self) -> None: ...
