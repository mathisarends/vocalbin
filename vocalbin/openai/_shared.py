from __future__ import annotations

from openai import AsyncOpenAI

from vocalbin.openai.credentials import Credentials
from vocalbin.ports import AsyncClosable


class _OpenAIClientOwner(AsyncClosable):
    def __init__(self, api_key: str | None, client: AsyncOpenAI | None) -> None:
        if api_key is not None and client is not None:
            raise ValueError("Pass either 'api_key' or 'client', not both.")
        if client is not None:
            self.client = client
        else:
            resolved_api_key = (
                api_key
                if api_key is not None
                else Credentials().api_key.get_secret_value()
            )
            self.client = AsyncOpenAI(api_key=resolved_api_key)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
