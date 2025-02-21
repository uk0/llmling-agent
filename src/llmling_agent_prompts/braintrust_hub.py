"""Braintrust prompt provider implementation."""

from __future__ import annotations

import os
from typing import Any

from braintrust import init_logger, load_prompt

from llmling_agent.prompts.base import BasePromptProvider
from llmling_agent_config.prompt_hubs import BraintrustConfig


class BraintrustPromptHub(BasePromptProvider):
    """Braintrust prompt provider implementation."""

    def __init__(self, config: BraintrustConfig):
        self.config = config or BraintrustConfig()
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        key = api_key or os.getenv("BRAINTRUST_API_KEY")
        init_logger(api_key=key, project=self.config.project)

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get and optionally compile a prompt from Braintrust."""
        import jinjarope

        env = jinjarope.Environment(enable_async=True)
        variables = variables or {}
        prompt = load_prompt(slug=name, version=version, project=self.config.project)
        assert prompt.prompt
        string = prompt.prompt.messages[0].content  # type: ignore
        assert isinstance(string, str)
        return await env.render_string_async(string, **variables)


if __name__ == "__main__":
    import asyncio

    from llmling_agent_config.prompt_hubs import BraintrustConfig

    config = BraintrustConfig(project="test")
    prompt_hub = BraintrustPromptHub(config)

    async def main():
        print(await prompt_hub.get_prompt("test-b4d4"))

    asyncio.run(main())
