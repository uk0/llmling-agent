"""Traceloop prompt provider implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from traceloop.sdk import Traceloop
from traceloop.sdk.prompts import get_prompt

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent_config.prompt_hubs import TraceloopConfig


class TraceloopPromptHub(BasePromptProvider):
    """Langfuse prompt provider implementation."""

    def __init__(self, config: TraceloopConfig):
        self.config = config
        api_key = (
            config.api_key.get_secret_value()
            if config.api_key
            else os.getenv("TRACELOOP_API_KEY")
        )
        Traceloop.init(traceloop_sync_enabled=True, api_key=api_key)

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get and optionally compile a prompt from Langfuse."""
        prompt = get_prompt(name, version=int(version) if version else None)
        return str(prompt)


if __name__ == "__main__":
    import asyncio

    from traceloop.sdk.prompts.client import PromptRegistryClient

    from llmling_agent_config.prompt_hubs import TraceloopConfig

    config = TraceloopConfig()
    prompt_hub = TraceloopPromptHub(config)
    print(PromptRegistryClient()._registry._prompts)

    async def main():
        print(await prompt_hub.get_prompt("test"))

    asyncio.run(main())
