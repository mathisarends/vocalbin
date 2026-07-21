from types import SimpleNamespace
from typing import Any, cast

from openai import AsyncOpenAI
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


async def test_injected_client_is_not_closed() -> None:
    fake_client = FakeClient()
    service = OpenAISpeechToText(client=cast(AsyncOpenAI, fake_client))

    await service.aclose()

    assert fake_client.closed is False
