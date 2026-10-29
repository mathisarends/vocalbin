from collections.abc import AsyncIterable, AsyncIterator
from typing import assert_type

from vocalbin import (
    RESET,
    AsyncClosable,
    IncrementalTextToSpeech,
    cartesia,
    deepgram,
    openai,
    piper,
)
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
    assert_type(
        await openai_tts.generate("Hallo", instructions=RESET),
        openai.TextToSpeechResponse,
    )
    async for chunk in openai_tts.stream("Hallo", instructions=RESET):
        assert_type(chunk, bytes)
    assert_type(await piper_tts.generate("Hallo"), piper.TextToSpeechResponse)


def realtime_calls(transcriber: realtime.Transcriber) -> None:
    assert_type(transcriber.stream(), AsyncIterator[realtime.events.TranscriptionEvent])


async def capability_calls(
    tts: IncrementalTextToSpeech[cartesia.TextToSpeechConfig, bytes],
    owned: AsyncClosable,
    text: AsyncIterable[str],
) -> None:
    async for chunk in tts.stream_incremental(text):
        assert_type(chunk, bytes)
    async with owned:
        await owned.aclose()


def provider_capabilities(tts: cartesia.TextToSpeech) -> None:
    incremental: IncrementalTextToSpeech[cartesia.TextToSpeechConfig, bytes] = tts
    closable: AsyncClosable = tts
    assert_type(
        incremental, IncrementalTextToSpeech[cartesia.TextToSpeechConfig, bytes]
    )
    assert_type(closable, AsyncClosable)


def microphone_calls() -> None:
    microphone = realtime.MicrophoneInput(chunk_duration_ms=40)
    assert_type(microphone.chunk_size, int)
    assert_type(microphone.chunk_duration_ms, float)
