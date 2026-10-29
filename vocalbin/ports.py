from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from enum import Enum
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel


class Reset(Enum):
    DEFAULT = "default"


RESET = Reset.DEFAULT


class AsyncClosable(ABC):
    @abstractmethod
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


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


class IncrementalTextToSpeech[ConfigT, EventT](StreamingTextToSpeech[ConfigT, EventT]):
    @abstractmethod
    def stream_incremental(
        self, text: AsyncIterable[str], *, config: ConfigT | None = None
    ) -> AsyncIterator[EventT]: ...


def resolve_config[ConfigT](
    config: ConfigT | None, default_config: ConfigT | None
) -> ConfigT:
    resolved = config if config is not None else default_config
    if resolved is None:
        raise ValueError("Provide configuration at construction time or per call.")
    return resolved


def merge_config[ConfigT: BaseModel](
    default_config: ConfigT | None,
    config_type: type[ConfigT],
    overrides: dict[str, Any],
) -> ConfigT:
    values = (
        default_config.model_dump(exclude_unset=True)
        if default_config is not None
        else {}
    )
    for key, value in overrides.items():
        if value is RESET:
            values.pop(key, None)
        elif value is not None:
            values[key] = value
    return config_type(**values)
