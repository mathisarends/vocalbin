from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from vocalbin.openai.models import (
    SpeechToTextRequest,
    SpeechToTextResponse,
)


class SpeechToText(ABC):
    @abstractmethod
    async def transcribe(
        self, request: SpeechToTextRequest
    ) -> SpeechToTextResponse: ...


class TextToSpeech[RequestT, ResponseT](ABC):
    @abstractmethod
    async def synthesize(self, request: RequestT) -> ResponseT: ...


class StreamingTextToSpeech[RequestT, EventT](ABC):
    @abstractmethod
    def stream(self, request: RequestT) -> AsyncIterator[EventT]: ...
