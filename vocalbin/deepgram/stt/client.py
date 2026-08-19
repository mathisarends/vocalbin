from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from vocalbin import ports
from vocalbin.deepgram.shared import DeepgramClientOwner
from vocalbin.deepgram.stt.models import (
    SpeechToTextConfig,
    SpeechToTextEncoding,
    SpeechToTextModel,
    SpeechToTextResponse,
)

if TYPE_CHECKING:
    from deepgram import AsyncDeepgramClient
    from deepgram.listen.v1.media import MediaTranscribeResponse


class SpeechToText(
    DeepgramClientOwner,
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
        client: AsyncDeepgramClient | None = None,
        model: SpeechToTextModel | str = SpeechToTextModel.NOVA_3,
        language: str | None = None,
        encoding: SpeechToTextEncoding | None = None,
        keyterms: list[str] | None = None,
        smart_format: bool | None = None,
        punctuate: bool | None = None,
        paragraphs: bool | None = None,
        utterances: bool | None = None,
        diarize: bool | None = None,
        filler_words: bool | None = None,
        numerals: bool | None = None,
        profanity_filter: bool | None = None,
        detect_language: bool | None = None,
    ) -> None:
        super().__init__(api_key, client)
        self.default_config = SpeechToTextConfig(
            model=model,
            language=language,
            encoding=encoding,
            keyterms=keyterms,
            smart_format=smart_format,
            punctuate=punctuate,
            paragraphs=paragraphs,
            utterances=utterances,
            diarize=diarize,
            filler_words=filler_words,
            numerals=numerals,
            profanity_filter=profanity_filter,
            detect_language=detect_language,
        )

    @overload
    async def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        model: SpeechToTextModel | str = SpeechToTextModel.NOVA_3,
        language: str | None = None,
        encoding: SpeechToTextEncoding | None = None,
        keyterms: list[str] | None = None,
        smart_format: bool | None = None,
        punctuate: bool | None = None,
        paragraphs: bool | None = None,
        utterances: bool | None = None,
        diarize: bool | None = None,
        filler_words: bool | None = None,
        numerals: bool | None = None,
        profanity_filter: bool | None = None,
        detect_language: bool | None = None,
    ) -> SpeechToTextResponse: ...

    @overload
    async def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        config: SpeechToTextConfig | None = None,
    ) -> SpeechToTextResponse: ...

    async def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        model: SpeechToTextModel | str | None = None,
        language: str | None = None,
        encoding: SpeechToTextEncoding | None = None,
        keyterms: list[str] | None = None,
        smart_format: bool | None = None,
        punctuate: bool | None = None,
        paragraphs: bool | None = None,
        utterances: bool | None = None,
        diarize: bool | None = None,
        filler_words: bool | None = None,
        numerals: bool | None = None,
        profanity_filter: bool | None = None,
        detect_language: bool | None = None,
        config: SpeechToTextConfig | None = None,
    ) -> SpeechToTextResponse:
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            language=language,
            encoding=encoding,
            keyterms=keyterms,
            smart_format=smart_format,
            punctuate=punctuate,
            paragraphs=paragraphs,
            utterances=utterances,
            diarize=diarize,
            filler_words=filler_words,
            numerals=numerals,
            profanity_filter=profanity_filter,
            detect_language=detect_language,
        )
        params = resolved_config.model_dump(
            exclude_none=True, mode="json", by_alias=True
        )
        result = await self.client.listen.v1.media.transcribe_file(
            request=_read_audio(audio), **params
        )
        return _to_response(result, resolved_config.model)


def _read_audio(audio: bytes | str | Path) -> bytes:
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("audio must not be empty")
        return audio
    audio_path = Path(audio)
    if not audio_path.is_file():
        raise ValueError(f"Audio file does not exist: {audio_path}")
    return audio_path.read_bytes()


def _to_response(
    result: MediaTranscribeResponse, model: SpeechToTextModel | str
) -> SpeechToTextResponse:
    raw = result.model_dump(mode="json")
    results = getattr(result, "results", None)
    if results is None:
        raise ValueError(
            "Deepgram returned an asynchronous callback response without a transcript."
        )
    channels = results.channels
    alternatives = channels[0].alternatives if channels else None
    alternative = alternatives[0] if alternatives else None
    return SpeechToTextResponse(
        text=(alternative.transcript or "") if alternative is not None else "",
        model=model,
        request_id=result.metadata.request_id,
        confidence=alternative.confidence if alternative is not None else None,
        detected_language=channels[0].detected_language if channels else None,
        raw=raw,
    )


def _resolve_call_config(
    *,
    config: SpeechToTextConfig | None,
    default_config: SpeechToTextConfig,
    model: SpeechToTextModel | str | None,
    language: str | None,
    encoding: SpeechToTextEncoding | None,
    keyterms: list[str] | None,
    smart_format: bool | None,
    punctuate: bool | None,
    paragraphs: bool | None,
    utterances: bool | None,
    diarize: bool | None,
    filler_words: bool | None,
    numerals: bool | None,
    profanity_filter: bool | None,
    detect_language: bool | None,
) -> SpeechToTextConfig:
    flat_values = (
        model,
        language,
        encoding,
        keyterms,
        smart_format,
        punctuate,
        paragraphs,
        utterances,
        diarize,
        filler_words,
        numerals,
        profanity_filter,
        detect_language,
    )
    has_flat_values = any(value is not None for value in flat_values)
    if config is not None:
        if has_flat_values:
            raise ValueError("Pass either 'config' or flat parameters, not both.")
        return config
    if not has_flat_values:
        return default_config

    values: dict[str, Any] = {
        "language": language,
        "encoding": encoding,
        "keyterms": keyterms,
        "smart_format": smart_format,
        "punctuate": punctuate,
        "paragraphs": paragraphs,
        "utterances": utterances,
        "diarize": diarize,
        "filler_words": filler_words,
        "numerals": numerals,
        "profanity_filter": profanity_filter,
        "detect_language": detect_language,
    }
    if model is not None:
        values["model"] = model
    return SpeechToTextConfig(**values)
