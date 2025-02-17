"""Webhook event source."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling_agent.messaging.events import EventData
from llmling_agent_events.base import EventSource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llmling_agent_config.events import WebhookConfig


class WebhookEventSource(EventSource):
    """Listens for webhook events on configured endpoint."""

    def __init__(self, config: WebhookConfig):
        from fastapi import FastAPI, Request

        self.config = config
        self.app = FastAPI()
        self._queue: asyncio.Queue[EventData] = asyncio.Queue()

        @self.app.post(config.path)
        async def handle_webhook(request: Request):
            # Verify signature if secret configured
            if self.config.secret:
                signature = request.headers.get("X-Hub-Signature")
                if not self._verify_signature(await request.body(), signature):
                    return {"status": "invalid signature"}

            # Process payload
            payload = await request.json()
            event = EventData.create(source=self.config.name, content=payload)
            await self._queue.put(event)
            return {"status": "ok"}

    async def connect(self):
        """Start webhook server."""
        import uvicorn

        self.server = uvicorn.Server(
            config=uvicorn.Config(
                self.app, host="0.0.0.0", port=self.config.port, log_level="error"
            )
        )
        await self.server.serve()

    async def disconnect(self):
        """Stop webhook server."""
        if self.server:
            await self.server.shutdown()

    async def events(self) -> AsyncGenerator[EventData, None]:
        """Yield events as they arrive."""
        while True:
            event = await self._queue.get()
            yield event

    def _verify_signature(self, payload: bytes, signature: str | None) -> bool:
        """Verify webhook signature."""
        import hashlib
        import hmac

        if not signature or not self.config.secret:
            return False
        key = self.config.secret.encode()
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()

        return hmac.compare_digest(signature, expected)
