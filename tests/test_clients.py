from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, omit
from openai.types.audio import Transcription

from vocalbin import (
    OpenAISpeechToText,
    OpenAITextToSpeech,
    SpeechToTextFormat,
    SpeechToTextRequest,
    TextToSpeechFormat,
    TextToSpeechRequest,
)


class FakeEndpoint:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, transcription: Any = None, speech: Any = None) -> None:
        self.transcriptions = FakeEndpoint(transcription)
        self.speech = FakeEndpoint(speech)
        self.audio = SimpleNamespace(
            transcriptions=self.transcriptions,
            speech=self.speech,
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_transcribe_bytes_returns_normalized_response() -> None:
    fake_client = FakeClient(transcription=Transcription(text="Hallo Welt"))
    service = OpenAISpeechToText(client=cast(AsyncOpenAI, fake_client))

    response = await service.transcribe(
        SpeechToTextRequest(audio=b"wave", filename="speech.wav", language="de")
    )

    assert response.text == "Hallo Welt"
    assert not isinstance(response.raw, str)
    assert response.raw["text"] == "Hallo Welt"
    call = fake_client.transcriptions.calls[0]
    assert call["file"] == ("speech.wav", b"wave")
    assert call["model"] == "gpt-4o-transcribe"
    assert call["language"] == "de"


async def test_transcribe_text_format_preserves_string_response() -> None:
    fake_client = FakeClient(transcription="plain transcript")
    service = OpenAISpeechToText(client=cast(AsyncOpenAI, fake_client))

    response = await service.transcribe(
        SpeechToTextRequest(audio=b"wave", response_format=SpeechToTextFormat.TEXT)
    )

    assert response.text == "plain transcript"
    assert response.raw == "plain transcript"


async def test_transcribe_reads_audio_from_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"wave")
    fake_client = FakeClient(transcription=Transcription(text="Hallo Datei"))
    service = OpenAISpeechToText(client=cast(AsyncOpenAI, fake_client))

    response = await service.transcribe(SpeechToTextRequest(audio_path=audio_path))

    assert response.text == "Hallo Datei"
    audio_file = fake_client.transcriptions.calls[0]["file"]
    assert audio_file.name == str(audio_path)
    assert audio_file.closed


async def test_synthesize_returns_audio_and_content_type() -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"generated-audio"))
    service = OpenAITextToSpeech(client=cast(AsyncOpenAI, fake_client))

    response = await service.synthesize(
        TextToSpeechRequest(
            text="Hallo",
            response_format=TextToSpeechFormat.WAV,
            speed=1.25,
        )
    )

    assert response.audio == b"generated-audio"
    assert response.content_type == "audio/wav"
    call = fake_client.speech.calls[0]
    assert call["input"] == "Hallo"
    assert call["response_format"] == "wav"
    assert call["speed"] == 1.25
    assert call["instructions"] is omit


async def test_synthesize_passes_instructions_and_omits_default_speed() -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"generated-audio"))
    service = OpenAITextToSpeech(client=cast(AsyncOpenAI, fake_client))

    await service.synthesize(TextToSpeechRequest(text="Hallo", instructions="Calm"))

    call = fake_client.speech.calls[0]
    assert call["instructions"] == "Calm"
    assert call["speed"] is omit


@pytest.mark.parametrize(
    ("response_format", "content_type"),
    [
        (TextToSpeechFormat.MP3, "audio/mpeg"),
        (TextToSpeechFormat.OPUS, "audio/opus"),
        (TextToSpeechFormat.AAC, "audio/aac"),
        (TextToSpeechFormat.FLAC, "audio/flac"),
        (TextToSpeechFormat.WAV, "audio/wav"),
        (TextToSpeechFormat.PCM, "audio/pcm"),
    ],
)
async def test_synthesize_maps_every_format_to_content_type(
    response_format: TextToSpeechFormat,
    content_type: str,
) -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"audio"))
    service = OpenAITextToSpeech(client=cast(AsyncOpenAI, fake_client))

    response = await service.synthesize(
        TextToSpeechRequest(text="Hallo", response_format=response_format)
    )

    assert response.content_type == content_type


def test_api_key_and_injected_client_are_mutually_exclusive() -> None:
    fake_client = FakeClient()

    with pytest.raises(ValueError, match="either 'api_key' or 'client'"):
        OpenAISpeechToText(
            api_key="explicit-key",
            client=cast(AsyncOpenAI, fake_client),
        )


async def test_owned_client_is_closed_by_context_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeClient()

    def create_client(*, api_key: str) -> AsyncOpenAI:
        assert api_key == "explicit-key"
        return cast(AsyncOpenAI, fake_client)

    monkeypatch.setattr("vocalbin.clients.AsyncOpenAI", create_client)

    async with OpenAISpeechToText(api_key="explicit-key") as service:
        assert service.client is fake_client

    assert fake_client.closed


async def test_injected_client_is_not_closed() -> None:
    fake_client = FakeClient()
    service = OpenAISpeechToText(client=cast(AsyncOpenAI, fake_client))

    await service.aclose()

    assert fake_client.closed is False
