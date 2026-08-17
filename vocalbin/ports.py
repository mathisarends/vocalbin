from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator


class WebSocketClient(ABC):
    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...


class SpeechToText[AudioT, ConfigT, ResponseT](ABC):
    @abstractmethod
    async def transcribe(
        self, audio: AudioT, *, config: ConfigT | None = None
    ) -> ResponseT: ...


class StreamingSpeechToText[ConfigT, EventT](ABC):
    @abstractmethod
    def stream(
        self,
        audio: AsyncIterable[bytes],
        *,
        config: ConfigT | None = None,
    ) -> AsyncIterator[EventT]: ...


class TextToSpeech[ConfigT, ResponseT](ABC):
    @abstractmethod
    async def generate(
        self, text: str, *, config: ConfigT | None = None
    ) -> ResponseT: ...


class StreamingTextToSpeech[ConfigT, EventT](ABC):
    @abstractmethod
    def stream(
        self, text: str, *, config: ConfigT | None = None
    ) -> AsyncIterator[EventT]: ...


def resolve_config[ConfigT](
    config: ConfigT | None, default_config: ConfigT | None
) -> ConfigT:
    resolved = config if config is not None else default_config
    if resolved is None:
        raise ValueError("Provide configuration at construction time or per call.")
    return resolved
