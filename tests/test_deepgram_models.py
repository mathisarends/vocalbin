import pytest

from vocalbin.deepgram import (
    AudioContainer,
    AudioEncoding,
    SpeechToTextConfig,
    SpeechToTextModel,
    StreamingSpeechToTextConfig,
    TextToSpeechConfig,
    TextToSpeechModel,
)


def test_deepgram_stt_config_defaults_to_nova_3() -> None:
    config = SpeechToTextConfig()

    assert config.model == SpeechToTextModel.NOVA_3
    assert config.model_dump(exclude_none=True, by_alias=True) == {"model": "nova-3"}


def test_deepgram_stt_config_serializes_keyterms_as_keyterm() -> None:
    config = SpeechToTextConfig(keyterms=["vocalbin"])

    assert config.model_dump(exclude_none=True, by_alias=True)["keyterm"] == [
        "vocalbin"
    ]


def test_deepgram_stt_config_rejects_blank_keyterms() -> None:
    with pytest.raises(ValueError, match="must not contain blank values"):
        SpeechToTextConfig(keyterms=[" "])


def test_deepgram_stt_config_rejects_keyterms_for_older_models() -> None:
    with pytest.raises(ValueError, match="only by nova-3"):
        SpeechToTextConfig(model=SpeechToTextModel.NOVA_2, keyterms=["vocalbin"])


def test_deepgram_stt_config_normalizes_language() -> None:
    assert SpeechToTextConfig(language=" de ").language == "de"


def test_deepgram_stt_config_rejects_blank_language() -> None:
    with pytest.raises(ValueError, match="language must not be blank"):
        SpeechToTextConfig(language="  ")


def test_deepgram_streaming_stt_config_defaults_to_flux() -> None:
    config = StreamingSpeechToTextConfig()

    assert config.model == "flux-general-en"
    assert config.encoding == "linear16"
    assert config.sample_rate == 16000


def test_deepgram_streaming_stt_config_rejects_eager_above_end_threshold() -> None:
    with pytest.raises(ValueError, match="eager_eot_threshold must not exceed"):
        StreamingSpeechToTextConfig(eager_eot_threshold=0.9, eot_threshold=0.5)


def test_deepgram_streaming_stt_config_compares_against_default_threshold() -> None:
    with pytest.raises(ValueError, match="eager_eot_threshold must not exceed"):
        StreamingSpeechToTextConfig(eager_eot_threshold=0.8)

    assert StreamingSpeechToTextConfig(eager_eot_threshold=0.7).eot_threshold is None


def test_deepgram_streaming_stt_config_rejects_blank_keyterms() -> None:
    with pytest.raises(ValueError, match="must not contain blank values"):
        StreamingSpeechToTextConfig(keyterms=[""])


def test_deepgram_streaming_stt_config_rejects_out_of_range_thresholds() -> None:
    with pytest.raises(ValueError):
        StreamingSpeechToTextConfig(eot_threshold=0.95)
    with pytest.raises(ValueError):
        StreamingSpeechToTextConfig(eot_timeout_ms=100)


def test_deepgram_tts_config_defaults_to_aura_2() -> None:
    config = TextToSpeechConfig()

    assert config.model == TextToSpeechModel.AURA_2_THALIA_EN
    assert config.encoding == AudioEncoding.LINEAR16
    assert config.content_type == "audio/pcm"


def test_deepgram_tts_config_reports_container_content_types() -> None:
    wav = TextToSpeechConfig(container=AudioContainer.WAV)
    ogg = TextToSpeechConfig(encoding=AudioEncoding.OPUS, container=AudioContainer.OGG)
    mp3 = TextToSpeechConfig(encoding=AudioEncoding.MP3, sample_rate=None)

    assert wav.content_type == "audio/wav"
    assert ogg.content_type == "audio/ogg"
    assert mp3.content_type == "audio/mpeg"


def test_deepgram_tts_config_rejects_bit_rate_for_uncompressed_audio() -> None:
    with pytest.raises(ValueError, match="bit_rate is not supported"):
        TextToSpeechConfig(encoding=AudioEncoding.LINEAR16, bit_rate=48000)


def test_deepgram_tts_config_rejects_wav_container_for_compressed_audio() -> None:
    with pytest.raises(ValueError, match="container 'wav' is not supported"):
        TextToSpeechConfig(encoding=AudioEncoding.MP3, container=AudioContainer.WAV)


def test_deepgram_tts_config_rejects_ogg_container_without_opus() -> None:
    with pytest.raises(ValueError, match="container 'ogg' requires encoding 'opus'"):
        TextToSpeechConfig(
            encoding=AudioEncoding.LINEAR16, container=AudioContainer.OGG
        )
