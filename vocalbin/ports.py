from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from vocalbin.openai.stt.models import (
    SpeechToTextRequest,
    SpeechToTextResponse,
)


class SpeechToText(ABC):
    @abstractmethod
    async def transcribe(
        self, request: SpeechToTextRequest
    ) -> SpeechToTextResponse: ...


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
        raise ValueError(
            "Provide 'config' either at construction time (default_config) or per call."
        )
    return resolved
