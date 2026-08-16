from typing import Any, cast, overload

from openai import AsyncOpenAI, omit

from vocalbin import ports
from vocalbin.openai._shared import _OpenAIClientOwner
from vocalbin.openai.tts.models import (
    TextToSpeechConfig,
    TextToSpeechFormat,
    TextToSpeechModel,
    TextToSpeechResponse,
    TextToSpeechVoice,
)

_MAX_TEXT_LENGTH = 4096

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class TextToSpeech(
    _OpenAIClientOwner,
    ports.TextToSpeech[TextToSpeechConfig, TextToSpeechResponse],
):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        model: TextToSpeechModel | str = TextToSpeechModel.GPT_4O_MINI_TTS,
        voice: TextToSpeechVoice | str = TextToSpeechVoice.MARIN,
        response_format: TextToSpeechFormat = TextToSpeechFormat.MP3,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = TextToSpeechConfig(
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            speed=speed,
        )

    @overload
    async def generate(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str = TextToSpeechModel.GPT_4O_MINI_TTS,
        voice: TextToSpeechVoice | str = TextToSpeechVoice.MARIN,
        response_format: TextToSpeechFormat = TextToSpeechFormat.MP3,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> TextToSpeechResponse: ...

    @overload
    async def generate(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> TextToSpeechResponse: ...

    async def generate(
        self,
        text: str,
        *,
        model: TextToSpeechModel | str | None = None,
        voice: TextToSpeechVoice | str | None = None,
        response_format: TextToSpeechFormat | None = None,
        instructions: str | None = None,
        speed: float | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> TextToSpeechResponse:
        text = _require_valid_text(text)
        resolved_config = _resolve_tts_config(
            config=config,
            default_config=self.default_config,
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            speed=speed,
        )
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


def _require_valid_text(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be blank")
    if len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(f"text must not exceed {_MAX_TEXT_LENGTH} characters")
    return text


def _resolve_tts_config(
    *,
    config: TextToSpeechConfig | None,
    default_config: TextToSpeechConfig | None,
    model: TextToSpeechModel | str | None,
    voice: TextToSpeechVoice | str | None,
    response_format: TextToSpeechFormat | None,
    instructions: str | None,
    speed: float | None,
) -> TextToSpeechConfig:
    flat_values = (model, voice, response_format, instructions, speed)
    has_flat_values = any(value is not None for value in flat_values)
    if config is not None:
        if has_flat_values:
            raise ValueError("Pass either 'config' or flat parameters, not both.")
        return config
    if not has_flat_values and default_config is not None:
        return default_config

    values: dict[str, Any] = {
        "instructions": instructions,
        "speed": speed,
    }
    if model is not None:
        values["model"] = model
    if voice is not None:
        values["voice"] = voice
    if response_format is not None:
        values["response_format"] = response_format
    return TextToSpeechConfig(**values)
