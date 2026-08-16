import inspect

import pytest

import vocalbin
from vocalbin import (
    SpeechToText as SpeechToTextPort,
)
from vocalbin import (
    StreamingSpeechToText,
    StreamingTextToSpeech,
)
from vocalbin import (
    TextToSpeech as TextToSpeechPort,
)
from vocalbin.cartesia import SpeechToText as CartesiaSpeechToText
from vocalbin.cartesia import TextToSpeech as CartesiaTextToSpeech
from vocalbin.openai import SpeechToText as OpenAISpeechToText
from vocalbin.openai import TextToSpeech as OpenAITextToSpeech
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
    assert inspect.isabstract(SpeechToTextPort)
    assert inspect.isabstract(TextToSpeechPort)
    assert inspect.isabstract(StreamingSpeechToText)
    assert inspect.isabstract(StreamingTextToSpeech)
    assert issubclass(OpenAISpeechToText, SpeechToTextPort)
    assert issubclass(OpenAITextToSpeech, TextToSpeechPort)
    assert issubclass(CartesiaTextToSpeech, TextToSpeechPort)
    assert issubclass(CartesiaTextToSpeech, StreamingTextToSpeech)
    assert issubclass(CartesiaSpeechToText, StreamingSpeechToText)

    with pytest.raises(TypeError):
        SpeechToTextPort()
    with pytest.raises(TypeError):
        TextToSpeechPort()
    with pytest.raises(TypeError):
        StreamingSpeechToText()
    with pytest.raises(TypeError):
        StreamingTextToSpeech()


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
