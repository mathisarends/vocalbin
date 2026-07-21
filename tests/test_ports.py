import inspect

import pytest

from vocalbin import OpenAISpeechToText, OpenAITextToSpeech, SpeechToText, TextToSpeech


def test_ports_are_abstract_and_openai_clients_implement_them() -> None:
    assert inspect.isabstract(SpeechToText)
    assert inspect.isabstract(TextToSpeech)
    assert issubclass(OpenAISpeechToText, SpeechToText)
    assert issubclass(OpenAITextToSpeech, TextToSpeech)

    with pytest.raises(TypeError):
        SpeechToText()
    with pytest.raises(TypeError):
        TextToSpeech()
