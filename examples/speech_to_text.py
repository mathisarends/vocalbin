"""Runnable speech-to-text examples covering every model and response format.

Run with a valid OPENAI_API_KEY:

    uv run python examples/speech_to_text.py

The examples need an audio file. If examples/output/sample.wav does not exist,
it is generated once via text-to-speech so every example below is self-contained.
"""

import asyncio
from pathlib import Path

from vocalbin import (
    OpenAISpeechToText,
    OpenAITextToSpeech,
    SpeechToTextFormat,
    SpeechToTextModel,
    SpeechToTextRequest,
    TextToSpeechFormat,
    TextToSpeechRequest,
    TimestampGranularity,
)

OUTPUT_DIR = Path(__file__).parent / "output"
SAMPLE = OUTPUT_DIR / "sample.wav"
SAMPLE_TEXT = "Guten Morgen, dies ist ein Test der Transkription mit vocalbin."


async def from_path() -> None:
    """Read audio from a file path (default model gpt-4o-transcribe, json)."""
    async with OpenAISpeechToText() as stt:
        response = await stt.transcribe(
            SpeechToTextRequest(audio_path=SAMPLE, language="de")
        )
    print(f"from_path: {response.text!r}")


async def from_bytes() -> None:
    """Read audio from raw bytes; `filename` only sets the multipart name."""
    audio = SAMPLE.read_bytes()
    async with OpenAISpeechToText() as stt:
        response = await stt.transcribe(
            SpeechToTextRequest(audio=audio, filename="sample.wav", language="de")
        )
    print(f"from_bytes: {response.text!r}")


async def gpt_transcribe_json_with_logprobs() -> None:
    """gpt-4o-transcribe: json + logprobs, plus prompt/temperature hints."""
    async with OpenAISpeechToText() as stt:
        response = await stt.transcribe(
            SpeechToTextRequest(
                audio_path=SAMPLE,
                model=SpeechToTextModel.GPT_4O_TRANSCRIBE,
                response_format=SpeechToTextFormat.JSON,
                language="de",
                prompt="vocalbin, Transkription",
                temperature=0.0,
                include=["logprobs"],
            )
        )
    print(f"gpt json+logprobs: {response.text!r}")


async def gpt_mini_text() -> None:
    """gpt-4o-mini-transcribe: text format returns the transcript as a plain string."""
    async with OpenAISpeechToText() as stt:
        response = await stt.transcribe(
            SpeechToTextRequest(
                audio_path=SAMPLE,
                model=SpeechToTextModel.GPT_4O_MINI_TRANSCRIBE,
                response_format=SpeechToTextFormat.TEXT,
            )
        )
    print(f"gpt-mini text: raw is str -> {isinstance(response.raw, str)}")


async def diarized() -> None:
    """gpt-4o-transcribe-diarize: diarized_json adds per-speaker segments in `raw`."""
    async with OpenAISpeechToText() as stt:
        response = await stt.transcribe(
            SpeechToTextRequest(
                audio_path=SAMPLE,
                model=SpeechToTextModel.GPT_4O_TRANSCRIBE_DIARIZE,
                response_format=SpeechToTextFormat.DIARIZED_JSON,
                language="de",
            )
        )
    print(f"diarized: {response.text!r}")


async def whisper_verbose_json_with_timestamps() -> None:
    """whisper-1: verbose_json is the only format that carries word/segment timestamps."""
    async with OpenAISpeechToText() as stt:
        response = await stt.transcribe(
            SpeechToTextRequest(
                audio_path=SAMPLE,
                model=SpeechToTextModel.WHISPER_1,
                response_format=SpeechToTextFormat.VERBOSE_JSON,
                timestamp_granularities=[
                    TimestampGranularity.WORD,
                    TimestampGranularity.SEGMENT,
                ],
            )
        )
    print(f"whisper verbose_json: {response.text!r}")


async def whisper_subtitle_formats() -> None:
    """whisper-1 additionally supports subtitle formats; text is the raw srt/vtt."""
    async with OpenAISpeechToText() as stt:
        for fmt in (SpeechToTextFormat.SRT, SpeechToTextFormat.VTT):
            response = await stt.transcribe(
                SpeechToTextRequest(
                    audio_path=SAMPLE,
                    model=SpeechToTextModel.WHISPER_1,
                    response_format=fmt,
                )
            )
            (OUTPUT_DIR / f"transcript.{fmt.value}").write_text(
                response.text, encoding="utf-8"
            )
            print(f"whisper {fmt.value}: {len(response.text)} chars")


async def _ensure_sample() -> None:
    if SAMPLE.exists():
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    async with OpenAITextToSpeech() as tts:
        response = await tts.synthesize(
            TextToSpeechRequest(
                text=SAMPLE_TEXT,
                response_format=TextToSpeechFormat.WAV,
            )
        )
    SAMPLE.write_bytes(response.audio)
    print(f"generated sample: {SAMPLE}")


async def main() -> None:
    await _ensure_sample()
    await from_path()
    await from_bytes()
    await gpt_transcribe_json_with_logprobs()
    await gpt_mini_text()
    await diarized()
    await whisper_verbose_json_with_timestamps()
    await whisper_subtitle_formats()


if __name__ == "__main__":
    asyncio.run(main())
