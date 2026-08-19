from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Self

from vocalbin.deepgram.credentials import Credentials

if TYPE_CHECKING:
    import httpx
    from deepgram import AsyncDeepgramClient


class DeepgramClientOwner:
    def __init__(self, api_key: str | None, client: AsyncDeepgramClient | None) -> None:
        if api_key is not None and client is not None:
            raise ValueError("Pass either 'api_key' or 'client', not both.")
        # The SDK exposes no close(); owning the transport keeps teardown explicit.
        self._http_client: httpx.AsyncClient | None = None
        if client is not None:
            self.client = client
        else:
            resolved_api_key = (
                api_key
                if api_key is not None
                else Credentials().api_key.get_secret_value()
            )
            self.client, self._http_client = _create_client(resolved_api_key)

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _create_client(api_key: str) -> tuple[AsyncDeepgramClient, httpx.AsyncClient]:
    try:
        import httpx
        from deepgram import AsyncDeepgramClient
    except ImportError as exc:
        raise ImportError("Deepgram support requires `vocalbin[deepgram]`.") from exc
    http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    return AsyncDeepgramClient(api_key=api_key, httpx_client=http_client), http_client
