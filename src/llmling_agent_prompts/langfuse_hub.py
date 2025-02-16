"""Langfuse prompt provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langfuse import Langfuse

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent_config.prompt_hubs import LangfuseConfig


class LangfusePromptHub(BasePromptProvider):
    """Langfuse prompt provider implementation."""

    def __init__(self, config: LangfuseConfig):
        self.config = config
        secret = config.secret_key.get_secret_value()
        pub = config.public_key.get_secret_value()
        self._client = Langfuse(secret_key=secret, public_key=pub, host=config.host)

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get and optionally compile a prompt from Langfuse."""
        prompt = self._client.get_prompt(
            name,
            type="text",
            version=int(version) if version else None,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
            max_retries=self.config.max_retries,
            fetch_timeout_seconds=self.config.fetch_timeout_seconds,
        )
        if variables:
            return prompt.compile(**variables)
        return prompt.prompt
