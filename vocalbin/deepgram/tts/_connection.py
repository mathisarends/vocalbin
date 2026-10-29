from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepgram import AsyncDeepgramClient
    from deepgram.speak.v1.socket_client import AsyncV1SocketClient


@dataclass
class _Lease:
    connection: AsyncV1SocketClient
    complete: bool = False


class _StreamingConnection:
    def __init__(self, client: AsyncDeepgramClient) -> None:
        self._client = client
        self._connection: AsyncV1SocketClient | None = None
        self._manager: AbstractAsyncContextManager[AsyncV1SocketClient] | None = None
        self._params: dict[str, str] | None = None
        self._stream_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._reuse = False
        self._closed = False

    async def connect(self, params: dict[str, str]) -> None:
        async with self._stream_lock:
            await self._get(params)
            self._reuse = True

    @asynccontextmanager
    async def acquire(self, params: dict[str, str]) -> AsyncIterator[_Lease]:
        # Speak v1 has no context IDs, so one socket carries one request at a time.
        async with self._stream_lock:
            lease = _Lease(await self._get(params))
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
