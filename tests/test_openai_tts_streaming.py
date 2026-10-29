import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from openai import AsyncOpenAI

from vocalbin import RESET
from vocalbin.openai import TextToSpeech, TextToSpeechConfig, TextToSpeechFormat


class GatedAudio(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"first"
        self.waiting.set()
        await self.release.wait()
        yield b"second"

    async def aclose(self) -> None:
        self.closed = True


async def test_openai_tts_stream_yields_before_the_response_finishes() -> None:
    audio = GatedAudio()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=audio)

    async with AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        tts = TextToSpeech(client=client, instructions="Calm", speed=1.2)
        stream = tts.stream(
            "Hallo", response_format=TextToSpeechFormat.PCM, instructions=RESET
        )
        try:
            assert await asyncio.wait_for(anext(stream), 10) == b"first"
            assert not audio.release.is_set()
            assert not audio.closed
            audio.release.set()
            assert [chunk async for chunk in stream] == [b"second"]
        finally:
            await stream.aclose()
        assert audio.closed
        body = json.loads(requests[0].content)
        assert body == {
            "input": "Hallo",
            "model": "gpt-4o-mini-tts",
            "voice": "marin",
            "response_format": "pcm",
            "speed": 1.2,
        }


@pytest.mark.parametrize("cancel", [False, True])
async def test_openai_tts_stream_closes_response_on_early_exit(cancel: bool) -> None:
    audio = GatedAudio()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=audio))
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        stream = TextToSpeech(client=client).stream(
            "Hallo", config=TextToSpeechConfig()
        )
        assert await anext(stream) == b"first"
        try:
            if cancel:
                task = asyncio.create_task(anext(stream))
                await audio.waiting.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await stream.aclose()
            assert audio.closed
        finally:
            audio.release.set()
            await stream.aclose()


@pytest.mark.parametrize("text", ["  ", "x" * 4097])
async def test_openai_tts_stream_validates_text_before_requesting(text: str) -> None:
    async with AsyncOpenAI(api_key="test") as client:
        with pytest.raises(ValueError):
            await anext(TextToSpeech(client=client).stream(text))
