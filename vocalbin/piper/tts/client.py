from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, overload

from vocalbin import ports
from vocalbin.piper.config import Config
from vocalbin.piper.tts.models import TextToSpeechConfig, TextToSpeechResponse

if TYPE_CHECKING:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from piper.voice import AudioChunk


class TextToSpeech(
    ports.TextToSpeech[TextToSpeechConfig, TextToSpeechResponse],
    ports.StreamingTextToSpeech[TextToSpeechConfig, bytes],
):
    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        *,
        voice: PiperVoice | None = None,
        default_config: TextToSpeechConfig | None = None,
    ) -> None:
        if model_path is not None and voice is not None:
            raise ValueError("Pass either 'model_path' or 'voice', not both.")
        if voice is None:
            if model_path is not None:
                resolved_model_path = Path(model_path)
                resolved_config_path = (
                    Path(config_path) if config_path is not None else None
                )
            else:
                config = Config()
                resolved_model_path = config.model_path
                resolved_config_path = config.config_path
            voice = _create_voice(resolved_model_path, resolved_config_path)
        self.voice = voice
        self.default_config = default_config

    @overload
    async def generate(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
    ) -> TextToSpeechResponse: ...

    @overload
    async def generate(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> TextToSpeechResponse: ...

    async def generate(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> TextToSpeechResponse:
        text = _require_non_blank_text(text)
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        audio = await asyncio.to_thread(
            lambda: b"".join(
                chunk.audio_int16_bytes
                for chunk in self._synthesize(text, resolved_config)
            )
        )
        return TextToSpeechResponse(
            audio=audio,
            sample_rate=self.voice.config.sample_rate,
            content_type="audio/pcm",
        )

    @overload
    def stream(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
    ) -> AsyncGenerator[bytes]: ...

    @overload
    def stream(
        self, text: str, *, config: TextToSpeechConfig | None = None
    ) -> AsyncGenerator[bytes]: ...

    async def stream(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
        config: TextToSpeechConfig | None = None,
    ) -> AsyncGenerator[bytes]:
        text = _require_non_blank_text(text)
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )
        chunks = (
            chunk.audio_int16_bytes for chunk in self._synthesize(text, resolved_config)
        )
        async for chunk in _stream_sync_generator(chunks):
            yield chunk

    def _synthesize(
        self, text: str, config: TextToSpeechConfig
    ) -> Iterator[AudioChunk]:
        return self.voice.synthesize(
            text,
            syn_config=_build_syn_config(config.to_piper_params()),
        )

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _create_voice(model_path: Path, config_path: Path | None) -> PiperVoice:
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise ImportError("Piper support requires `vocalbin[piper]`.") from exc
    return PiperVoice.load(
        str(model_path),
        config_path=str(config_path) if config_path is not None else None,
    )


def _require_non_blank_text(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be blank")
    return text


def _build_syn_config(params: dict[str, Any]) -> SynthesisConfig:
    try:
        from piper.config import SynthesisConfig
    except ImportError as exc:
        raise ImportError("Piper support requires `vocalbin[piper]`.") from exc
    return SynthesisConfig(**params)


async def _stream_sync_generator(generator: Iterator[bytes]) -> AsyncGenerator[bytes]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    def worker() -> None:
        try:
            for chunk in generator:
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except BaseException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        thread.join()


def _resolve_call_config(
    *,
    config: TextToSpeechConfig | None,
    default_config: TextToSpeechConfig | None,
    speaker_id: int | None,
    length_scale: float | None,
    noise_scale: float | None,
    noise_w_scale: float | None,
) -> TextToSpeechConfig:
    flat_values = (speaker_id, length_scale, noise_scale, noise_w_scale)
    has_flat_values = any(value is not None for value in flat_values)
    if config is not None:
        if has_flat_values:
            raise ValueError("Pass either 'config' or flat parameters, not both.")
        return config
    if not has_flat_values and default_config is not None:
        return default_config
    return TextToSpeechConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )
