from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from vocalbin import ports
from vocalbin.deepgram.shared import DeepgramClientOwner
from vocalbin.deepgram.stt.models import (
    SpeechToTextConfig,
    SpeechToTextEncoding,
    SpeechToTextModel,
    SpeechToTextResponse,
)
from vocalbin.ports import Reset, merge_config

if TYPE_CHECKING:
    from deepgram import AsyncDeepgramClient, ListenV1Response
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
        sample_rate: int | None = None,
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
            sample_rate=sample_rate,
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
        model: SpeechToTextModel | str | Reset = SpeechToTextModel.NOVA_3,
        language: str | None | Reset = None,
        encoding: SpeechToTextEncoding | None | Reset = None,
        sample_rate: int | None | Reset = None,
        keyterms: list[str] | None | Reset = None,
        smart_format: bool | None | Reset = None,
        punctuate: bool | None | Reset = None,
        paragraphs: bool | None | Reset = None,
        utterances: bool | None | Reset = None,
        diarize: bool | None | Reset = None,
        filler_words: bool | None | Reset = None,
        numerals: bool | None | Reset = None,
        profanity_filter: bool | None | Reset = None,
        detect_language: bool | None | Reset = None,
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
        model: SpeechToTextModel | str | None | Reset = None,
        language: str | None | Reset = None,
        encoding: SpeechToTextEncoding | None | Reset = None,
        sample_rate: int | None | Reset = None,
        keyterms: list[str] | None | Reset = None,
        smart_format: bool | None | Reset = None,
        punctuate: bool | None | Reset = None,
        paragraphs: bool | None | Reset = None,
        utterances: bool | None | Reset = None,
        diarize: bool | None | Reset = None,
        filler_words: bool | None | Reset = None,
        numerals: bool | None | Reset = None,
        profanity_filter: bool | None | Reset = None,
        detect_language: bool | None | Reset = None,
        config: SpeechToTextConfig | None = None,
    ) -> SpeechToTextResponse:
        resolved_config = _resolve_call_config(
            config=config,
            default_config=self.default_config,
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
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
    result = cast("ListenV1Response", result)
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
    model: SpeechToTextModel | str | None | Reset,
    language: str | None | Reset,
    encoding: SpeechToTextEncoding | None | Reset,
    sample_rate: int | None | Reset,
    keyterms: list[str] | None | Reset,
    smart_format: bool | None | Reset,
    punctuate: bool | None | Reset,
    paragraphs: bool | None | Reset,
    utterances: bool | None | Reset,
    diarize: bool | None | Reset,
    filler_words: bool | None | Reset,
    numerals: bool | None | Reset,
    profanity_filter: bool | None | Reset,
    detect_language: bool | None | Reset,
) -> SpeechToTextConfig:
    flat_values = (
        model,
        language,
        encoding,
        sample_rate,
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
        "sample_rate": sample_rate,
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
    return merge_config(default_config, SpeechToTextConfig, values)
