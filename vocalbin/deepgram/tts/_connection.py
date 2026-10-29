from __future__ import annotations

import asyncio
from asyncio import sleep
from collections import deque
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepgram import AsyncDeepgramClient
    from deepgram.speak.v1.socket_client import AsyncV1SocketClient


class _FlushLimiter:
    def __init__(self) -> None:
        self._sent: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            now = monotonic()
            while self._sent and now - self._sent[0] >= 60:
                self._sent.popleft()
            if len(self._sent) < 20:
                self._sent.append(now)
                return
            await sleep(60 - (now - self._sent[0]))


@dataclass
class _Lease:
    connection: AsyncV1SocketClient
    limiter: _FlushLimiter
    complete: bool = False
    pending_flushes: int = 0

    async def flush(self, message: Any) -> None:
        await self.limiter.acquire()
        # A response can arrive while send_control() is still awaiting the socket.
        self.pending_flushes += 1
        await self.connection.send_control(message)


class _StreamingConnection:
    def __init__(self, client: AsyncDeepgramClient) -> None:
        self._client = client
        self._connection: AsyncV1SocketClient | None = None
        self._manager: AbstractAsyncContextManager[AsyncV1SocketClient] | None = None
        self._params: dict[str, str] | None = None
        self._stream_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._reuse = False
        self._flush_limiter = _FlushLimiter()
        self._closed = False

    async def connect(self, params: dict[str, str]) -> None:
        async with self._stream_lock:
            await self._get(params)
            self._reuse = True

    @asynccontextmanager
    async def acquire(self, params: dict[str, str]) -> AsyncIterator[_Lease]:
        # Speak v1 has no context IDs, so one socket carries one request at a time.
        async with self._stream_lock:
            lease = _Lease(await self._get(params), self._flush_limiter)
            try:
                yield lease
            finally:
                if not self._reuse or not lease.complete:
                    async with self._lifecycle_lock:
                        await self._disconnect()

    async def _get(self, params: dict[str, str]) -> AsyncV1SocketClient:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("TextToSpeech has been closed.")
            if self._connection is None or params != self._params:
                await self._disconnect()
                manager = self._client.speak.v1.connect(**params)
                connection = await manager.__aenter__()
                self._manager = manager
                self._connection = connection
                self._params = params.copy()
            return self._connection

    async def _disconnect(self) -> None:
        manager = self._manager
        self._manager = None
        self._connection = None
        self._params = None
        if manager is not None:
            await manager.__aexit__(None, None, None)

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
            await self._disconnect()
