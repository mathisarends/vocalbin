from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, omit
from openai.types.audio import Transcription

from vocalbin.openai import (
    SpeechToText,
    SpeechToTextConfig,
    SpeechToTextFormat,
    SpeechToTextModel,
    TextToSpeech,
    TextToSpeechConfig,
    TextToSpeechFormat,
    TextToSpeechModel,
    TextToSpeechVoice,
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
    service = SpeechToText(client=cast(AsyncOpenAI, fake_client))

    response = await service.transcribe(
        b"wave",
        filename="speech.wav",
        model=SpeechToTextModel.GPT_4O_TRANSCRIBE,
        language="de",
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
    service = SpeechToText(client=cast(AsyncOpenAI, fake_client))

    response = await service.transcribe(
        b"wave",
        response_format=SpeechToTextFormat.TEXT,
    )

    assert response.text == "plain transcript"
    assert response.raw == "plain transcript"


async def test_transcribe_reads_audio_from_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"wave")
    fake_client = FakeClient(transcription=Transcription(text="Hallo Datei"))
    service = SpeechToText(client=cast(AsyncOpenAI, fake_client))

    response = await service.transcribe(audio_path)

    assert response.text == "Hallo Datei"
    audio_file = fake_client.transcriptions.calls[0]["file"]
    assert audio_file.name == str(audio_path)
    assert audio_file.closed


async def test_transcribe_accepts_config_and_constructor_defaults() -> None:
    fake_client = FakeClient(transcription="plain transcript")
    config = SpeechToTextConfig(response_format=SpeechToTextFormat.TEXT)
    service = SpeechToText(
        client=cast(AsyncOpenAI, fake_client),
        response_format=SpeechToTextFormat.TEXT,
    )

    default_response = await service.transcribe(b"first")
    config_response = await service.transcribe(b"second", config=config)

    assert default_response.response_format == SpeechToTextFormat.TEXT
    assert config_response.response_format == SpeechToTextFormat.TEXT


async def test_transcribe_rejects_config_with_flat_parameters() -> None:
    service = SpeechToText(client=cast(AsyncOpenAI, FakeClient()))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await cast(Any, service).transcribe(
            b"wave",
            language="de",
            config=SpeechToTextConfig(),
        )


@pytest.mark.parametrize(
    ("audio", "filename", "message"),
    [
        (b"", "speech.wav", "audio must not be empty"),
        (b"wave", "  ", "filename must not be blank"),
    ],
)
async def test_transcribe_rejects_invalid_byte_input(
    audio: bytes,
    filename: str,
    message: str,
) -> None:
    service = SpeechToText(client=cast(AsyncOpenAI, FakeClient()))

    with pytest.raises(ValueError, match=message):
        await service.transcribe(audio, filename=filename)


async def test_transcribe_rejects_invalid_paths(tmp_path: Path) -> None:
    service = SpeechToText(client=cast(AsyncOpenAI, FakeClient()))

    with pytest.raises(ValueError, match="does not exist"):
        await service.transcribe(tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="not a file"):
        await service.transcribe(str(tmp_path))


async def test_generate_returns_audio_and_content_type() -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"generated-audio"))
    service = TextToSpeech(client=cast(AsyncOpenAI, fake_client))

    response = await service.generate(
        "Hallo",
        model=TextToSpeechModel.GPT_4O_MINI_TTS,
        voice=TextToSpeechVoice.CEDAR,
        response_format=TextToSpeechFormat.WAV,
        speed=1.25,
    )

    assert response.audio == b"generated-audio"
    assert response.content_type == "audio/wav"
    call = fake_client.speech.calls[0]
    assert call["input"] == "Hallo"
    assert call["response_format"] == "wav"
    assert call["speed"] == 1.25
    assert call["instructions"] is omit


async def test_generate_passes_instructions_and_omits_default_speed() -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"generated-audio"))
    service = TextToSpeech(client=cast(AsyncOpenAI, fake_client))

    await service.generate("Hallo", instructions="Calm")

    call = fake_client.speech.calls[0]
    assert call["instructions"] == "Calm"
    assert call["speed"] is omit


async def test_generate_uses_constructor_defaults() -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"generated-audio"))
    service = TextToSpeech(
        client=cast(AsyncOpenAI, fake_client),
        response_format=TextToSpeechFormat.WAV,
    )

    response = await service.generate("Hallo")

    assert response.content_type == "audio/wav"


async def test_generate_uses_builtin_defaults() -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"generated-audio"))
    service = TextToSpeech(client=cast(AsyncOpenAI, fake_client))

    response = await service.generate("Hallo")

    assert response.model == TextToSpeechModel.GPT_4O_MINI_TTS
    assert response.voice == TextToSpeechVoice.MARIN
    assert response.response_format == TextToSpeechFormat.MP3


async def test_generate_rejects_config_with_flat_parameters() -> None:
    service = TextToSpeech(client=cast(AsyncOpenAI, FakeClient()))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await cast(Any, service).generate(
            "Hallo",
            voice=TextToSpeechVoice.CEDAR,
            config=TextToSpeechConfig(),
        )


async def test_generate_rejects_blank_text() -> None:
    service = TextToSpeech(client=cast(AsyncOpenAI, FakeClient()))

    with pytest.raises(ValueError, match="text must not be blank"):
        await service.generate("   ", config=TextToSpeechConfig())


async def test_generate_rejects_text_over_max_length() -> None:
    service = TextToSpeech(client=cast(AsyncOpenAI, FakeClient()))

    with pytest.raises(ValueError, match="must not exceed 4096 characters"):
        await service.generate("x" * 4097, config=TextToSpeechConfig())


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
async def test_generate_maps_every_format_to_content_type(
    response_format: TextToSpeechFormat,
    content_type: str,
) -> None:
    fake_client = FakeClient(speech=SimpleNamespace(content=b"audio"))
    service = TextToSpeech(client=cast(AsyncOpenAI, fake_client))

    response = await service.generate(
        "Hallo", config=TextToSpeechConfig(response_format=response_format)
    )

    assert response.content_type == content_type


def test_api_key_and_injected_client_are_mutually_exclusive() -> None:
    fake_client = FakeClient()

    with pytest.raises(ValueError, match="either 'api_key' or 'client'"):
        SpeechToText(
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

    monkeypatch.setattr("vocalbin.openai._shared.AsyncOpenAI", create_client)

    async with SpeechToText(api_key="explicit-key") as service:
        assert service.client is fake_client

    assert fake_client.closed


async def test_injected_client_is_not_closed() -> None:
    fake_client = FakeClient()
    service = SpeechToText(client=cast(AsyncOpenAI, fake_client))

    await service.aclose()

    assert fake_client.closed is False
