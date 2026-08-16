import asyncio
import logging
import os
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from vocalbin.cartesia import (
    RawOutputFormat,
    SpeechToText,
    TextToSpeech,
    events,
)

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
MOCK_LLM_FIRST_TOKEN_DELAY = 0.6
MOCK_LLM_TOKEN_DELAY = 0.12


class ElapsedFormatter(logging.Formatter):
    def __init__(self, started_at: float) -> None:
        super().__init__("[+%(elapsed)7.1f ms] %(message)s")
        self.started_at = started_at

    def format(self, record: logging.LogRecord) -> str:
        record.elapsed = (perf_counter() - self.started_at) * 1000
        return super().format(record)


def configure_logging(started_at: float) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter(started_at))
    logger = logging.getLogger("cartesia-round-trip")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


async def microphone_audio(logger: logging.Logger) -> AsyncIterator[bytes]:
    try:
        import sounddevice
    except ImportError as exc:
        raise ImportError(
            "This example requires sounddevice. Run it with the 'audio' extra."
        ) from exc

    stream = sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=1,
        dtype="int16",
    )
    stream.start()
    logger.info("Microphone started; speak one English sentence")
    try:
        while True:
            data, overflowed = await asyncio.to_thread(stream.read, CHUNK_SIZE)
            if overflowed:
                logger.warning("Microphone input overflow")
            yield bytes(data)
    finally:
        stream.stop()
        stream.close()
        logger.info("Microphone stopped")


async def transcribe_turn(logger: logging.Logger) -> str:
    transcript = ""
    async with SpeechToText() as speech_to_text:
        event_stream = speech_to_text.stream(microphone_audio(logger))
        try:
            async for event in event_stream:
                match event:
                    case events.Connected():
                        logger.info("STT connected")
                    case events.TurnStart():
                        logger.info("Speech detected")
                    case events.TurnUpdate(transcript=text):
                        transcript = text
                        logger.info("STT partial: %r", text)
                    case events.TurnEagerEnd(transcript=text):
                        transcript = text
                        logger.info("STT eager end: %r", text)
                    case events.TurnResume():
                        logger.info("Speech resumed")
                    case events.TurnEnd(transcript=text):
                        transcript = text
                        logger.info("STT finished: %r", text)
                        break
        finally:
            await event_stream.aclose()

    if not transcript.strip():
        raise RuntimeError("Cartesia finished the turn without a transcript")
    return transcript


async def mock_llm(transcript: str, logger: logging.Logger) -> AsyncIterator[str]:
    logger.info("Mock LLM started")
    await asyncio.sleep(MOCK_LLM_FIRST_TOKEN_DELAY)
    chunks = (
        "I heard you say: ",
        f'"{transcript}". ',
        "This response came from a mocked language model.",
    )
    for index, chunk in enumerate(chunks):
        if index:
            await asyncio.sleep(MOCK_LLM_TOKEN_DELAY)
        logger.info("Mock LLM chunk %d/%d: %r", index + 1, len(chunks), chunk)
        yield chunk
    logger.info("Mock LLM finished")


async def speak_response(
    transcript: str, voice_id: str, logger: logging.Logger
) -> None:
    try:
        import sounddevice
    except ImportError as exc:
        raise ImportError(
            "This example requires sounddevice. Run it with the 'audio' extra."
        ) from exc

    output_format = RawOutputFormat(sample_rate=SAMPLE_RATE)
    output: Any = sounddevice.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    first_audio = True

    logger.info("TTS started")
    output.start()
    try:
        async with TextToSpeech() as text_to_speech:
            async for audio in text_to_speech.stream_incremental(
                mock_llm(transcript, logger),
                voice_id=voice_id,
                language="en",
                output_format=output_format,
            ):
                if first_audio:
                    logger.info("TTS first audio received")
                    first_audio = False
                await asyncio.to_thread(output.write, audio)
        logger.info("TTS stream finished")
    finally:
        await asyncio.to_thread(output.stop)
        output.close()
        logger.info("Audio playback finished")


async def main() -> None:
    started_at = perf_counter()
    logger = configure_logging(started_at)
    voice_id = os.environ["CARTESIA_VOICE_ID"]

    logger.info("Round trip started")
    transcript = await transcribe_turn(logger)
    await speak_response(transcript, voice_id, logger)
    logger.info("Round trip finished")


if __name__ == "__main__":
    asyncio.run(main())
