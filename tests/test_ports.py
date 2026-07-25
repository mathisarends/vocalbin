import inspect

import pytest

import vocalbin
from vocalbin import OpenAISpeechToText, OpenAITextToSpeech, SpeechToText, TextToSpeech
from vocalbin.realtime import (
    AudioInput,
    MicrophoneInput,
    OpenAIRealtimeProvider,
    OpenAIRealtimeTranscriber,
    OpenAIRealtimeTranslator,
    RealtimeProvider,
    RealtimeTranscription,
    RealtimeTranslation,
)


def test_ports_are_abstract_and_openai_clients_implement_them() -> None:
    assert inspect.isabstract(SpeechToText)
    assert inspect.isabstract(TextToSpeech)
    assert issubclass(OpenAISpeechToText, SpeechToText)
    assert issubclass(OpenAITextToSpeech, TextToSpeech)

    with pytest.raises(TypeError):
        SpeechToText()
    with pytest.raises(TypeError):
        TextToSpeech()


def test_realtime_ports_are_abstract_and_openai_services_implement_them() -> None:
    assert inspect.isabstract(AudioInput)
    assert inspect.isabstract(RealtimeProvider)
    assert inspect.isabstract(RealtimeTranscription)
    assert inspect.isabstract(RealtimeTranslation)
    assert issubclass(MicrophoneInput, AudioInput)
    assert issubclass(OpenAIRealtimeProvider, RealtimeProvider)
    assert issubclass(OpenAIRealtimeTranscriber, RealtimeTranscription)
    assert issubclass(OpenAIRealtimeTranslator, RealtimeTranslation)

    for port in (
        AudioInput,
        RealtimeProvider,
        RealtimeTranscription,
        RealtimeTranslation,
    ):
        with pytest.raises(TypeError):
            port()


def test_root_api_does_not_flatten_realtime_namespaces() -> None:
    assert not hasattr(vocalbin, "OpenAIRealtimeTranscriber")
    assert not hasattr(vocalbin, "OpenAIRealtimeTranslator")
    assert not hasattr(vocalbin, "RealtimeTranslationConfig")
