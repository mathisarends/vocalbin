from collections.abc import AsyncIterable, AsyncIterator
from typing import assert_type

from vocalbin import cartesia, deepgram, openai, piper
from vocalbin.openai import realtime


async def speech_calls(
    cartesia_tts: cartesia.TextToSpeech,
    deepgram_stt: deepgram.SpeechToText,
    openai_tts: openai.TextToSpeech,
    piper_tts: piper.TextToSpeech,
    text: AsyncIterable[str],
) -> None:
    assert_type(
        await cartesia_tts.generate("Hallo", speed=1.1), cartesia.TextToSpeechResponse
    )
    async for chunk in cartesia_tts.stream("Hallo", speed=1.1):
        assert_type(chunk, bytes)
    async for chunk in cartesia_tts.stream_incremental(text, speed=1.1):
        assert_type(chunk, bytes)
    assert_type(await deepgram_stt.transcribe(b"audio"), deepgram.SpeechToTextResponse)
    assert_type(await openai_tts.generate("Hallo"), openai.TextToSpeechResponse)
    assert_type(await piper_tts.generate("Hallo"), piper.TextToSpeechResponse)


def realtime_calls(transcriber: realtime.Transcriber) -> None:
    assert_type(transcriber.stream(), AsyncIterator[realtime.events.TranscriptionEvent])
