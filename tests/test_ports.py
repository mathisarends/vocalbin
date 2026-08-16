import inspect

import pytest

import vocalbin
from vocalbin import cartesia, openai, piper, ports
from vocalbin.openai.realtime import (
    AudioInput,
    MicrophoneInput,
    Provider,
    RealtimeProvider,
    RealtimeTranscription,
    RealtimeTranslation,
    Transcriber,
    Translator,
)


def test_ports_are_abstract_and_openai_clients_implement_them() -> None:
    assert inspect.isabstract(ports.SpeechToText)
    assert inspect.isabstract(ports.TextToSpeech)
    assert inspect.isabstract(ports.StreamingSpeechToText)
    assert inspect.isabstract(ports.StreamingTextToSpeech)
    assert issubclass(openai.SpeechToText, ports.SpeechToText)
    assert issubclass(openai.TextToSpeech, ports.TextToSpeech)
    assert issubclass(cartesia.TextToSpeech, ports.TextToSpeech)
    assert issubclass(cartesia.TextToSpeech, ports.StreamingTextToSpeech)
    assert issubclass(cartesia.SpeechToText, ports.StreamingSpeechToText)
    assert issubclass(piper.TextToSpeech, ports.TextToSpeech)
    assert issubclass(piper.TextToSpeech, ports.StreamingTextToSpeech)

    with pytest.raises(TypeError):
        ports.SpeechToText()
    with pytest.raises(TypeError):
        ports.TextToSpeech()
    with pytest.raises(TypeError):
        ports.StreamingSpeechToText()
    with pytest.raises(TypeError):
        ports.StreamingTextToSpeech()


def test_realtime_ports_are_abstract_and_openai_services_implement_them() -> None:
    assert inspect.isabstract(AudioInput)
    assert inspect.isabstract(RealtimeProvider)
    assert inspect.isabstract(RealtimeTranscription)
    assert inspect.isabstract(RealtimeTranslation)
    assert issubclass(MicrophoneInput, AudioInput)
    assert issubclass(Provider, RealtimeProvider)
    assert issubclass(Transcriber, RealtimeTranscription)
    assert issubclass(Translator, RealtimeTranslation)

    for port in (
        AudioInput,
        RealtimeProvider,
        RealtimeTranscription,
        RealtimeTranslation,
    ):
        with pytest.raises(TypeError):
            port()


def test_root_api_does_not_flatten_realtime_namespaces() -> None:
    assert not hasattr(vocalbin, "Transcriber")
    assert not hasattr(vocalbin, "Translator")
    assert not hasattr(vocalbin, "RealtimeTranslationConfig")
