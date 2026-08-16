import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vocalbin.piper import TextToSpeech, TextToSpeechConfig
from vocalbin.piper.tts import client as piper_clients


class FakeVoiceConfig:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate


class FakeAudioChunk:
    def __init__(self, audio_int16_bytes: bytes) -> None:
        self.audio_int16_bytes = audio_int16_bytes


class FakeVoice:
    def __init__(
        self, chunks: list[bytes] | None = None, error: Exception | None = None
    ) -> None:
        self.config = FakeVoiceConfig(22050)
        self.chunks = chunks if chunks is not None else [b"one", b"two"]
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, text: str, syn_config: Any = None):
        self.calls.append({"text": text, "syn_config": syn_config})
        for chunk in self.chunks:
            yield FakeAudioChunk(chunk)
        if self.error is not None:
            raise self.error


async def collect(stream):
    return [chunk async for chunk in stream]


def stub_syn_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(piper_clients, "_build_syn_config", lambda params: params)


async def test_generate_returns_joined_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_syn_config(monkeypatch)
    voice = FakeVoice([b"one", b"two"])
    service = TextToSpeech(voice=cast(Any, voice))

    response = await service.generate(
        "Hallo",
        speaker_id=1,
        length_scale=1.1,
        noise_scale=0.5,
        noise_w_scale=0.6,
    )

    assert response.audio == b"onetwo"
    assert response.sample_rate == 22050
    assert response.content_type == "audio/pcm"
    assert voice.calls == [
        {
            "text": "Hallo",
            "syn_config": {
                "speaker_id": 1,
                "length_scale": 1.1,
                "noise_scale": 0.5,
                "noise_w_scale": 0.6,
            },
        }
    ]


async def test_generate_uses_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_syn_config(monkeypatch)
    voice = FakeVoice([b"one", b"two"])
    service = TextToSpeech(voice=cast(Any, voice), speaker_id=2)

    response = await service.generate("Hallo")

    assert response.audio == b"onetwo"
    assert voice.calls == [{"text": "Hallo", "syn_config": {"speaker_id": 2}}]


async def test_generate_uses_builtin_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_syn_config(monkeypatch)
    voice = FakeVoice()
    service = TextToSpeech(voice=cast(Any, voice))

    await service.generate("Hallo")

    assert voice.calls == [{"text": "Hallo", "syn_config": {}}]


async def test_generate_rejects_config_with_flat_parameters() -> None:
    service = TextToSpeech(voice=cast(Any, FakeVoice()))

    with pytest.raises(ValueError, match="either 'config' or flat parameters"):
        await cast(Any, service).generate(
            "Hallo",
            speaker_id=1,
            config=TextToSpeechConfig(),
        )


async def test_generate_rejects_blank_text() -> None:
    service = TextToSpeech(voice=cast(Any, FakeVoice()))

    with pytest.raises(ValueError, match="text must not be blank"):
        await service.generate("   ", config=TextToSpeechConfig())


async def test_stream_yields_chunks_from_background_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_syn_config(monkeypatch)
    voice = FakeVoice([b"a", b"b", b"c"])
    service = TextToSpeech(voice=cast(Any, voice))

    assert await collect(service.stream("Hallo", speaker_id=2)) == [b"a", b"b", b"c"]
    assert voice.calls == [{"text": "Hallo", "syn_config": {"speaker_id": 2}}]


async def test_stream_propagates_generator_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_syn_config(monkeypatch)
    voice = FakeVoice([b"a"], error=RuntimeError("synthesis failed"))
    service = TextToSpeech(voice=cast(Any, voice))
    config = TextToSpeechConfig()

    with pytest.raises(RuntimeError, match="synthesis failed"):
        await collect(service.stream("Hallo", config=config))


def test_model_path_and_voice_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either 'model_path' or 'voice'"):
        TextToSpeech(model_path="voice.onnx", voice=cast(Any, FakeVoice()))


async def test_context_manager_returns_self_and_closes_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_syn_config(monkeypatch)
    voice = FakeVoice([b"audio"])

    async with TextToSpeech(voice=cast(Any, voice)) as service:
        response = await service.generate("Hallo", config=TextToSpeechConfig())

    assert response.audio == b"audio"


def test_create_voice_uses_official_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "piper":
            return SimpleNamespace(
                PiperVoice=SimpleNamespace(
                    load=lambda model_path, config_path=None: SimpleNamespace(
                        model_path=model_path, config_path=config_path
                    )
                )
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    voice = piper_clients._create_voice(Path("voice.onnx"), Path("voice.onnx.json"))

    assert voice.model_path == "voice.onnx"
    assert voice.config_path == "voice.onnx.json"


def test_create_voice_explains_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "piper":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"vocalbin\[piper\]"):
        piper_clients._create_voice(Path("voice.onnx"), None)


def test_build_syn_config_uses_official_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "piper.config":
            return SimpleNamespace(
                SynthesisConfig=lambda **params: SimpleNamespace(**params)
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    syn_config = piper_clients._build_syn_config({"speaker_id": 1})

    assert syn_config.speaker_id == 1


def test_build_syn_config_explains_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "piper.config":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"vocalbin\[piper\]"):
        piper_clients._build_syn_config({})


async def test_uses_environment_model_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    voice = FakeVoice([b"audio"])
    monkeypatch.setenv("PIPER_MODEL_PATH", "voices/de.onnx")
    monkeypatch.delenv("PIPER_CONFIG_PATH", raising=False)

    def create_voice(model_path: Path, config_path: Path | None) -> Any:
        assert model_path == Path("voices/de.onnx")
        assert config_path is None
        return voice

    monkeypatch.setattr(piper_clients, "_create_voice", create_voice)

    service = TextToSpeech()

    assert service.voice is voice


async def test_explicit_model_path_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice = FakeVoice([b"audio"])
    monkeypatch.setenv("PIPER_MODEL_PATH", "voices/de.onnx")

    def create_voice(model_path: Path, config_path: Path | None) -> Any:
        assert model_path == Path("voices/en.onnx")
        assert config_path == Path("voices/en.onnx.json")
        return voice

    monkeypatch.setattr(piper_clients, "_create_voice", create_voice)

    service = TextToSpeech(
        model_path="voices/en.onnx", config_path="voices/en.onnx.json"
    )

    assert service.voice is voice
