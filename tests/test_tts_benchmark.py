import asyncio
from collections.abc import AsyncGenerator, AsyncIterable

import pytest

from scripts import benchmark_tts as bench
from vocalbin import AsyncClosable, Connectable


class FakeService(AsyncClosable, Connectable):
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.streams = 0
        self.stream_closed = False

    async def connect(self) -> None:
        self.connected = True

    async def aclose(self) -> None:
        self.closed = True

    async def stream(self, text: str) -> AsyncGenerator[bytes]:
        self.streams += 1
        try:
            yield b""
            yield text.encode()
        finally:
            self.stream_closed = True

    async def stream_incremental(
        self, text: AsyncIterable[str]
    ) -> AsyncGenerator[bytes]:
        self.streams += 1
        try:
            async for chunk in text:
                yield chunk.encode()
        finally:
            self.stream_closed = True


async def test_benchmark_times_the_first_audio_and_drains_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((10.0, 10.1, 10.3))
    monkeypatch.setattr(bench, "perf_counter", lambda: next(ticks))
    service = FakeService()
    sample = await bench.measure(service, bench.Options("fake", text="audio"))
    assert sample.first_audio_ms == pytest.approx(100)
    assert sample.total_ms == pytest.approx(300)
    assert sample.audio_bytes == 5
    assert sample.text_to_first_audio_ms == sample.first_audio_ms
    assert service.stream_closed


async def test_benchmark_reports_text_availability_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((10.0, 10.05, 10.08, 10.4))
    monkeypatch.setattr(bench, "perf_counter", lambda: next(ticks))
    sample = await bench.measure(
        FakeService(),
        bench.Options(
            "fake",
            text_chunks=("First", "Second"),
            first_text_delay_ms=1,
            chunk_delay_ms=1,
        ),
    )
    assert sample.first_text_ms == pytest.approx(50)
    assert sample.text_to_first_audio_ms == pytest.approx(30)
    assert sample.audio_bytes == len(b"FirstSecond")


async def test_benchmark_separates_new_clients_and_preconnected_reuse() -> None:
    clients: list[FakeService] = []

    def build() -> FakeService:
        service = FakeService()
        clients.append(service)
        return service

    report = await bench.benchmark(bench.Options("fake", runs=3, warmup=2), build)
    assert len(clients) == 4
    assert all(service.closed for service in clients)
    assert [service.streams for service in clients] == [1, 1, 1, 5]
    assert clients[-1].connected
    assert report["phases"]["warm"]["connect_ms"] is not None
    assert report["phases"]["cold"]["connect_ms"] is None
    assert len(report["phases"]["warm"]["samples"]) == 3


@pytest.mark.parametrize("mode", ["cold", "warm"])
async def test_benchmark_closes_clients_after_failed_measurements(mode: str) -> None:
    class BrokenService(FakeService):
        async def stream(self, text: str) -> AsyncGenerator[bytes]:
            try:
                raise RuntimeError("failed")
                yield b"unreachable"
            finally:
                self.stream_closed = True

    service = BrokenService()
    with pytest.raises(RuntimeError, match="failed"):
        await bench.benchmark(bench.Options("fake", mode=mode, runs=1), lambda: service)
    assert service.closed
    assert service.stream_closed


async def test_benchmark_rejects_empty_audio_and_times_out_with_cleanup() -> None:
    class EmptyService(FakeService):
        async def stream(self, text: str) -> AsyncGenerator[bytes]:
            if False:
                yield b""

    with pytest.raises(ValueError, match="without audio"):
        await bench.measure(EmptyService(), bench.Options("fake"))

    class StalledService(FakeService):
        async def stream(self, text: str) -> AsyncGenerator[bytes]:
            try:
                await asyncio.Event().wait()
                yield b"unreachable"
            finally:
                self.stream_closed = True

    service = StalledService()
    with pytest.raises(TimeoutError):
        await bench.measure(service, bench.Options("fake", timeout_seconds=0.01))
    assert service.stream_closed


def test_benchmark_summary_uses_median_and_nearest_rank_p95() -> None:
    samples = [bench.Sample(float(i), 0, float(i), 20, 5) for i in range(1, 21)]
    summary = bench.summarize(samples)
    assert summary["p50_first_audio_ms"] == 10.5
    assert summary["p95_first_audio_ms"] == 19


@pytest.mark.parametrize(
    "extra",
    [
        ["--runs", "0"],
        ["--warmup", "0"],
        ["--timeout-seconds", "nan"],
        ["--text", " "],
        ["--text-chunks", "Hello"],
        ["--first-text-delay-ms", "1"],
        ["--buffer-delay-ms", "5001"],
    ],
)
def test_benchmark_rejects_invalid_options(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        bench.parse_options(["--provider", "openai", *extra])


def test_benchmark_parses_incremental_text_without_loading_credentials() -> None:
    options = bench.parse_options(
        [
            "--provider",
            "deepgram",
            "--text-chunks",
            "First. ",
            "Second.",
            "--chunk-delay-ms",
            "200",
        ]
    )
    assert options.text_chunks == ("First. ", "Second.")
    assert options.text == "First. Second."
    assert options.flush_on == "first_chunk"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "response_format"),
        ("cartesia", "voice_id"),
        ("deepgram", "flush_on"),
        ("piper", "model_path"),
    ],
)
def test_benchmark_builds_the_selected_provider_with_streaming_settings(
    monkeypatch: pytest.MonkeyPatch, provider: str, expected: str
) -> None:
    import importlib

    module = importlib.import_module(f"vocalbin.{provider}")
    calls = []
    monkeypatch.setattr(
        module, "TextToSpeech", lambda **kwargs: calls.append(kwargs) or kwargs
    )
    options = bench.Options(
        provider,
        model="test" if provider != "piper" else None,
        voice="voice-id",
        model_path="voice.onnx",
    )
    settings = bench.create_service(options)
    assert expected in settings
    assert len(calls) == 1


async def test_benchmark_produces_text_while_connection_setup_is_waiting() -> None:
    class ConnectingService(FakeService):
        async def stream_incremental(
            self, text: AsyncIterable[str]
        ) -> AsyncGenerator[bytes]:
            await asyncio.sleep(0.02)
            async for chunk in text:
                yield chunk.encode()

    sample = await bench.measure(
        ConnectingService(), bench.Options("fake", text_chunks=("Hi",))
    )
    assert sample.first_text_ms < sample.first_audio_ms
    assert sample.text_to_first_audio_ms > 0


async def test_benchmark_paces_flushes_outside_the_measured_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pauses: list[float] = []

    async def pause(delay: float) -> None:
        pauses.append(delay)

    monkeypatch.setattr(bench, "sleep", pause)
    options = bench.Options(
        "deepgram", mode="warm", runs=2, warmup=1, text_chunks=("First.", "Second.")
    )
    report = await bench.benchmark(options, FakeService)
    assert report["request_pause_ms"] == 6200
    assert pauses == [6.2, 6.2]


def test_benchmark_pacing_can_be_disabled_for_burst_measurements() -> None:
    assert bench.request_pause_ms(bench.Options("deepgram", pause_ms=0)) == 0
    assert bench.request_pause_ms(bench.Options("deepgram", flush_on="end")) == 3100
    assert (
        bench.request_pause_ms(
            bench.Options("deepgram", flush_on="chunk", text_chunks=("a", "b", "c"))
        )
        == 9300
    )
