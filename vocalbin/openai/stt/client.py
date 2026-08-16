from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.audio import TranscriptionCreateResponse

from vocalbin import ports
from vocalbin.openai._shared import _OpenAIClientOwner
from vocalbin.openai.stt.models import SpeechToTextRequest, SpeechToTextResponse

type TranscriptionResult = TranscriptionCreateResponse | str


class SpeechToText(
    _OpenAIClientOwner,
    ports.SpeechToText[SpeechToTextRequest, SpeechToTextResponse],
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(api_key, client)

    async def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        params = request.model_dump(
            exclude={"audio_path", "audio", "filename"},
            exclude_none=True,
            mode="json",
        )

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
