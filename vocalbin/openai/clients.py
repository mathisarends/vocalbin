from types import TracebackType
from typing import Any, Self, cast

from openai import AsyncOpenAI, omit
from openai.types.audio import TranscriptionCreateResponse

from vocalbin.openai.credentials import OpenAICredentials
from vocalbin.openai.models import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechConfig,
    TextToSpeechResponse,
)
from vocalbin.ports import SpeechToText, TextToSpeech, resolve_config

_MAX_TEXT_LENGTH = 4096

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
        if client is not None:
            self.client = client
        else:
            resolved_api_key = (
                api_key
                if api_key is not None
                else OpenAICredentials().api_key.get_secret_value()
            )
            self.client = AsyncOpenAI(api_key=resolved_api_key)
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


class OpenAITextToSpeech(
    _OpenAIClientOwner,
    TextToSpeech[TextToSpeechConfig, TextToSpeechResponse],
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        default_config: TextToSpeechConfig | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = default_config

    async def generate(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> TextToSpeechResponse:
        text = _require_valid_text(text)
        resolved_config = resolve_config(config, self.default_config)
        result = await self.client.audio.speech.create(
            input=text,
            model=resolved_config.model,
            voice=resolved_config.voice,
            instructions=resolved_config.instructions
            if resolved_config.instructions is not None
            else omit,
            response_format=cast(Any, resolved_config.response_format),
            speed=resolved_config.speed if resolved_config.speed is not None else omit,
        )

        return TextToSpeechResponse(
            audio=result.content,
            model=resolved_config.model,
            voice=resolved_config.voice,
            response_format=resolved_config.response_format,
            content_type=_CONTENT_TYPES[resolved_config.response_format],
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
                    await self.client.audio.transcriptions.create(
                        file=audio_file, **params
                    ),
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


def _require_valid_text(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be blank")
    if len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(f"text must not exceed {_MAX_TEXT_LENGTH} characters")
    return text
