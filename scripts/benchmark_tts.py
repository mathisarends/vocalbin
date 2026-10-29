from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
from asyncio import sleep
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import asdict, dataclass
from importlib.metadata import version
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

from vocalbin import Connectable

if TYPE_CHECKING:
    from vocalbin.deepgram import FlushMode

_DEFAULT_TEXT = "Hello! This is a first-audio latency benchmark."


@dataclass(frozen=True)
class Options:
    provider: str
    mode: str = "both"
    runs: int = 10
    warmup: int = 1
    text: str = _DEFAULT_TEXT
    text_chunks: tuple[str, ...] = ()
    first_text_delay_ms: int = 0
    chunk_delay_ms: int = 0
    model: str | None = None
    voice: str | None = None
    model_path: str | None = None
    buffer_delay_ms: int = 0
    flush_on: FlushMode = "first_chunk"
    timeout_seconds: float = 30
    pause_ms: int | None = None


@dataclass(frozen=True)
class Sample:
    first_audio_ms: float
    first_text_ms: float
    text_to_first_audio_ms: float
    total_ms: float
    audio_bytes: int


def create_service(options: Options) -> Any:
    kwargs: dict[str, Any] = {}
    if options.model is not None:
        kwargs["model"] = options.model
    if options.provider == "openai":
        from vocalbin.openai import TextToSpeech as OpenAITTS
        from vocalbin.openai import TextToSpeechFormat

        if options.voice is not None:
            kwargs["voice"] = options.voice
        return OpenAITTS(response_format=TextToSpeechFormat.PCM, **kwargs)
    if options.provider == "cartesia":
        from vocalbin.cartesia import RawOutputFormat
        from vocalbin.cartesia import TextToSpeech as CartesiaTTS

        return CartesiaTTS(
            voice_id=options.voice,
            output_format=RawOutputFormat(),
            max_buffer_delay_ms=options.buffer_delay_ms,
            **kwargs,
        )
    if options.provider == "deepgram":
        from vocalbin.deepgram import TextToSpeech as DeepgramTTS

        return DeepgramTTS(flush_on=options.flush_on, **kwargs)
    from vocalbin.piper import TextToSpeech as PiperTTS

    return PiperTTS(model_path=options.model_path)


async def measure(service: Any, options: Options) -> Sample:
    first_text_ms = 0.0
    first_audio_ms: float | None = None
    audio_bytes = 0
    started = perf_counter()

    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=1)

    async def produce_text() -> None:
        nonlocal first_text_ms
        if options.first_text_delay_ms:
            await asyncio.sleep(options.first_text_delay_ms / 1000)
        for index, chunk in enumerate(options.text_chunks):
            if index:
                await asyncio.sleep(options.chunk_delay_ms / 1000)
            else:
                first_text_ms = (perf_counter() - started) * 1000
            await queue.put(chunk)
        await queue.put(None)

    async def text() -> AsyncIterator[str]:
        while (chunk := await queue.get()) is not None:
            yield chunk

    producer = asyncio.create_task(produce_text()) if options.text_chunks else None
    try:
        stream = (
            service.stream_incremental(text())
            if options.text_chunks
            else service.stream(options.text)
        )
        async with asyncio.timeout(options.timeout_seconds):
            async with aclosing(stream):
                async for chunk in stream:
                    if not chunk:
                        continue
                    if first_audio_ms is None:
                        first_audio_ms = (perf_counter() - started) * 1000
                    audio_bytes += len(chunk)
    finally:
        if producer is not None:
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
    total_ms = (perf_counter() - started) * 1000
    if first_audio_ms is None:
        raise ValueError(
            "The stream ended without audio; no latency sample was recorded."
        )
    return Sample(
        first_audio_ms,
        first_text_ms,
        first_audio_ms - first_text_ms,
        total_ms,
        audio_bytes,
    )


def summarize(samples: list[Sample]) -> dict[str, float]:
    values = sorted(sample.first_audio_ms for sample in samples)
    return {
        "p50_first_audio_ms": median(values),
        "p95_first_audio_ms": values[math.ceil(len(values) * 0.95) - 1],
        "min_first_audio_ms": values[0],
        "max_first_audio_ms": values[-1],
    }


def request_pause_ms(options: Options) -> int:
    if options.pause_ms is not None:
        return options.pause_ms
    if options.provider != "deepgram":
        return 0
    chunks = len(options.text_chunks) or 1
    flushes = (
        chunks
        if options.flush_on == "chunk"
        else min(chunks, 2)
        if options.flush_on == "first_chunk"
        else 1
    )
    return 3100 * flushes


async def benchmark(
    options: Options, factory: Callable[[], Any] | None = None
) -> dict[str, Any]:
    build = factory if factory is not None else lambda: create_service(options)
    phases: dict[str, Any] = {}
    pause_ms = request_pause_ms(options)
    measured_once = False

    async def take_sample(service: Any) -> Sample:
        nonlocal measured_once
        if measured_once and pause_ms:
            await sleep(pause_ms / 1000)
        measured_once = True
        return await measure(service, options)

    resolved_config: dict[str, Any] | None = None
    for mode in ("cold", "warm") if options.mode == "both" else (options.mode,):
        samples: list[Sample] = []
        setup_ms: list[float] = []
        connect_ms: float | None = None
        service = None
        try:
            if mode == "warm":
                started = perf_counter()
                service = build()
                setup_ms.append((perf_counter() - started) * 1000)
                if isinstance(service, Connectable):
                    started = perf_counter()
                    async with asyncio.timeout(options.timeout_seconds):
                        await service.connect()
                    connect_ms = (perf_counter() - started) * 1000
                for _ in range(options.warmup):
                    await take_sample(service)
            for _ in range(options.runs):
                if mode == "cold":
                    started = perf_counter()
                    service = build()
                    setup_ms.append((perf_counter() - started) * 1000)
                try:
                    samples.append(await take_sample(service))
                    config = getattr(service, "default_config", None)
                    if config is not None:
                        resolved_config = config.model_dump(mode="json")
                finally:
                    if mode == "cold" and service is not None:
                        closing = service
                        service = None
                        await closing.aclose()
        finally:
            if service is not None:
                await service.aclose()
        phases[mode] = {
            "summary": summarize(samples),
            "samples": [asdict(sample) for sample in samples],
            "setup_ms": setup_ms,
            "connect_ms": connect_ms,
            "warmup_requests": options.warmup if mode == "warm" else 0,
        }
    return {
        "options": asdict(options),
        "request_pause_ms": pause_ms,
        "resolved_config": resolved_config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vocalbin": version("vocalbin"),
        },
        "phases": phases,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def parse_options(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        description=(
            "Measure TTS first-audio latency with full stream draining. "
            "Selected remote providers make billable API requests."
        )
    )
    parser.add_argument(
        "--provider", required=True, choices=("openai", "cartesia", "deepgram", "piper")
    )
    parser.add_argument("--mode", choices=("cold", "warm", "both"), default="both")
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--warmup", type=positive_int, default=1)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", default=None)
    source.add_argument("--text-chunks", nargs="+")
    parser.add_argument("--first-text-delay-ms", type=nonnegative_int, default=0)
    parser.add_argument("--chunk-delay-ms", type=nonnegative_int, default=0)
    parser.add_argument("--pause-ms", type=nonnegative_int, default=None)
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--model-path")
    parser.add_argument("--buffer-delay-ms", type=nonnegative_int, default=0)
    parser.add_argument(
        "--flush-on", choices=("end", "first_chunk", "chunk"), default="first_chunk"
    )
    parser.add_argument("--timeout-seconds", type=positive_float, default=30)
    args = parser.parse_args(argv)
    chunks = tuple(args.text_chunks or ())
    text = args.text if args.text is not None else _DEFAULT_TEXT
    if not text.strip() or any(not chunk.strip() for chunk in chunks):
        parser.error("text and text chunks must not be blank")
    if args.provider == "cartesia" and not args.voice:
        parser.error("--voice is required for Cartesia")
    if chunks and args.provider not in ("cartesia", "deepgram"):
        parser.error("--text-chunks requires Cartesia or Deepgram")
    if (args.first_text_delay_ms or args.chunk_delay_ms) and not chunks:
        parser.error("text delays require --text-chunks")
    if args.provider == "piper" and args.model:
        parser.error("use --model-path for Piper")
    if args.provider != "piper" and args.model_path:
        parser.error("--model-path requires Piper")
    if args.buffer_delay_ms > 5000:
        parser.error("--buffer-delay-ms must be between 0 and 5000")
    values = vars(args)
    values["text"] = "".join(chunks) if chunks else text
    values["text_chunks"] = chunks
    return Options(**values)


def main(argv: list[str] | None = None) -> None:
    options = parse_options(argv)
    print(json.dumps(asyncio.run(benchmark(options)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
