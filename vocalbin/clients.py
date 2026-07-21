from types import TracebackType
from typing import Any, Self, cast

from openai import AsyncOpenAI, omit
from openai.types.audio import TranscriptionCreateResponse

from vocalbin.interfaces import SpeechToText, TextToSpeech
from vocalbin.models import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

type TranscriptionResult = TranscriptionCreateResponse | str


class _OpenAIClientOwner:
    def __init__(self, api_key: str | None, client: AsyncOpenAI | None) -> None:
        if api_key is not None and client is not None:
            raise ValueError("Pass either 'api_key' or 'client', not both.")
        self.client = client if client is not None else AsyncOpenAI(api_key=api_key)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


class OpenAITextToSpeech(_OpenAIClientOwner, TextToSpeech):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(api_key, client)

    async def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        result = await self.client.audio.speech.create(
            input=request.text,
            model=request.model,
            voice=request.voice,
            instructions=request.instructions if request.instructions is not None else omit,
            response_format=request.response_format,
            speed=request.speed if request.speed is not None else omit,
        )

        return TextToSpeechResponse(
            audio=result.content,
            model=request.model,
            voice=request.voice,
            response_format=request.response_format,
            content_type=_CONTENT_TYPES[request.response_format],
        )


class OpenAISpeechToText(_OpenAIClientOwner, SpeechToText):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(api_key, client)

    async def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        params = request.to_openai_params()

        if request.audio is not None:
            file = (request.filename, request.audio)
            result = cast(
                TranscriptionResult,
                await self.client.audio.transcriptions.create(file=file, **params),
            )
        else:
            audio_path = request.audio_path
            assert audio_path is not None
            with audio_path.open("rb") as audio_file:
                result = cast(
                    TranscriptionResult,
                    await self.client.audio.transcriptions.create(file=audio_file, **params),
                )

        return SpeechToTextResponse(
            text=_extract_text(result),
            model=request.model,
            response_format=request.response_format,
            raw=_serialize_result(result),
        )


def _extract_text(result: TranscriptionResult) -> str:
    if isinstance(result, str):
        return result
    return result.text


def _serialize_result(result: TranscriptionResult) -> dict[str, Any] | str:
    if isinstance(result, str):
        return result
    return result.model_dump(mode="python")
