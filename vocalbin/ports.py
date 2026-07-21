from abc import ABC, abstractmethod

from vocalbin.models import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)


class SpeechToText(ABC):
    @abstractmethod
    async def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse: ...


class TextToSpeech(ABC):
    @abstractmethod
    async def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResponse: ...
