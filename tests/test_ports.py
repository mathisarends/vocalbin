import inspect

import pytest

import vocalbin
from vocalbin import cartesia, openai, piper, ports
from vocalbin.openai import realtime


def test_ports_are_abstract_and_openai_clients_implement_them() -> None:
    assert inspect.isabstract(ports.SpeechToText)
    assert inspect.isabstract(ports.TextToSpeech)
    assert inspect.isabstract(ports.StreamingSpeechToText)
    assert inspect.isabstract(ports.StreamingTextToSpeech)
    assert inspect.isabstract(ports.WebSocketClient)
    assert issubclass(openai.SpeechToText, ports.SpeechToText)
    assert issubclass(openai.TextToSpeech, ports.TextToSpeech)
    assert issubclass(cartesia.TextToSpeech, ports.TextToSpeech)
    assert issubclass(cartesia.TextToSpeech, ports.StreamingTextToSpeech)
    assert issubclass(cartesia.SpeechToText, ports.StreamingSpeechToText)
    assert issubclass(cartesia.TextToSpeech, ports.WebSocketClient)
    assert issubclass(cartesia.SpeechToText, ports.WebSocketClient)
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
    with pytest.raises(TypeError):
        ports.WebSocketClient()


def test_realtime_ports_are_abstract_and_openai_services_implement_them() -> None:
    assert inspect.isabstract(realtime.ports.AudioInput)
    assert inspect.isabstract(realtime.ports.Provider)
    assert inspect.isabstract(realtime.ports.Transcription)
    assert inspect.isabstract(realtime.ports.Translation)
    assert inspect.isabstract(realtime.ports.WebSocketSession)
    assert issubclass(realtime.MicrophoneInput, realtime.ports.AudioInput)
    assert issubclass(realtime.Provider, realtime.ports.Provider)
    assert issubclass(realtime.Transcriber, realtime.ports.Transcription)
    assert issubclass(realtime.Translator, realtime.ports.Translation)

    for port in (
        realtime.ports.AudioInput,
        realtime.ports.Provider,
        realtime.ports.Transcription,
        realtime.ports.Translation,
        realtime.ports.WebSocketSession,
    ):
        with pytest.raises(TypeError):
            port()


def test_root_api_does_not_flatten_realtime_namespaces() -> None:
    assert not hasattr(vocalbin, "Transcriber")
    assert not hasattr(vocalbin, "Translator")
    assert not hasattr(vocalbin, "TranslationConfig")
