from pathlib import Path
from typing import Any, Literal, cast, overload

from openai import AsyncOpenAI
from openai.types.audio import TranscriptionCreateResponse

from vocalbin import ports
from vocalbin.openai._shared import _OpenAIClientOwner
from vocalbin.openai.stt.models import (
    SpeechToTextConfig,
    SpeechToTextFormat,
    SpeechToTextModel,
    SpeechToTextResponse,
    TimestampGranularity,
)

type TranscriptionResult = TranscriptionCreateResponse | str


class SpeechToText(
    _OpenAIClientOwner,
    ports.SpeechToText[
        bytes | str | Path,
        SpeechToTextConfig,
        SpeechToTextResponse,
    ],
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        model: SpeechToTextModel | str = SpeechToTextModel.GPT_4O_TRANSCRIBE,
        response_format: SpeechToTextFormat = SpeechToTextFormat.JSON,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[TimestampGranularity] | None = None,
        include: list[Literal["logprobs"]] | None = None,
        chunking_strategy: Literal["auto"] | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = SpeechToTextConfig(
            model=model,
            response_format=response_format,
            language=language,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            include=include,
            chunking_strategy=chunking_strategy,
            extra_body=extra_body,
        )

    @overload
    async def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        filename: str = "utterance.wav",
        model: SpeechToTextModel | str = SpeechToTextModel.GPT_4O_TRANSCRIBE,
        response_format: SpeechToTextFormat = SpeechToTextFormat.JSON,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[TimestampGranularity] | None = None,
        include: list[Literal["logprobs"]] | None = None,
        chunking_strategy: Literal["auto"] | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> SpeechToTextResponse: ...

    @overload
    async def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        filename: str = "utterance.wav",
        config: SpeechToTextConfig | None = None,
    ) -> SpeechToTextResponse: ...

    async def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        filename: str = "utterance.wav",
        model: SpeechToTextModel | str | None = None,
        response_format: SpeechToTextFormat | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[TimestampGranularity] | None = None,
        include: list[Literal["logprobs"]] | None = None,
        chunking_strategy: Literal["auto"] | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        config: SpeechToTextConfig | None = None,
    ) -> SpeechToTextResponse:
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            response_format=response_format,
            language=language,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            include=include,
            chunking_strategy=chunking_strategy,
            extra_body=extra_body,
        )
        params = resolved_config.model_dump(exclude_none=True, mode="json")

        if isinstance(audio, bytes):
            if not audio:
                raise ValueError("audio must not be empty")
            if not filename.strip():
                raise ValueError("filename must not be blank")
            result = cast(
                TranscriptionResult,
                await self.client.audio.transcriptions.create(
                    file=(filename, audio),
                    **params,
                ),
            )
        else:
            audio_path = Path(audio)
            if not audio_path.exists():
                raise ValueError(f"Audio file does not exist: {audio_path}")
            if not audio_path.is_file():
                raise ValueError(f"Audio path is not a file: {audio_path}")
            with audio_path.open("rb") as audio_file:
                result = cast(
                    TranscriptionResult,
                    await self.client.audio.transcriptions.create(
                        file=audio_file, **params
                    ),
                )

        return SpeechToTextResponse(
            text=_extract_text(result),
            model=resolved_config.model,
            response_format=resolved_config.response_format,
            raw=_serialize_result(result),
        )


def _resolve_call_config(
    *,
    config: SpeechToTextConfig | None,
    default_config: SpeechToTextConfig | None,
    model: SpeechToTextModel | str | None,
    response_format: SpeechToTextFormat | None,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    timestamp_granularities: list[TimestampGranularity] | None,
    include: list[Literal["logprobs"]] | None,
    chunking_strategy: Literal["auto"] | dict[str, Any] | None,
    extra_body: dict[str, Any] | None,
) -> SpeechToTextConfig:
    flat_values = (
        model,
        response_format,
        language,
        prompt,
        temperature,
        timestamp_granularities,
        include,
        chunking_strategy,
        extra_body,
    )
    has_flat_values = any(value is not None for value in flat_values)
    if config is not None:
        if has_flat_values:
            raise ValueError("Pass either 'config' or flat parameters, not both.")
        return config
    if not has_flat_values and default_config is not None:
        return default_config

    values: dict[str, Any] = {
        "language": language,
        "prompt": prompt,
        "temperature": temperature,
        "timestamp_granularities": timestamp_granularities,
        "include": include,
        "chunking_strategy": chunking_strategy,
        "extra_body": extra_body,
    }
    if model is not None:
        values["model"] = model
    if response_format is not None:
        values["response_format"] = response_format
    return SpeechToTextConfig(**values)


def _extract_text(result: TranscriptionResult) -> str:
    if isinstance(result, str):
        return result
    return result.text


def _serialize_result(result: TranscriptionResult) -> dict[str, Any] | str:
    if isinstance(result, str):
        return result
    return result.model_dump(mode="python")
