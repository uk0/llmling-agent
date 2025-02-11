from __future__ import annotations

import os
from typing import Any

from braintrust import init_logger, load_prompt

from llmling_agent.models.prompt_hubs import BraintrustConfig
from llmling_agent.prompts.base import BasePromptProvider


class BraintrustPromptHub(BasePromptProvider):
    """Braintrust prompt provider implementation."""

    def __init__(self, config: BraintrustConfig):
        self.config = config or BraintrustConfig()
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        init_logger(api_key=api_key or os.getenv("BRAINTRUST_API_KEY"))

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get and optionally compile a prompt from Braintrust."""
        variables = variables or {}
        prompt = load_prompt(name, version=version)
        return prompt.build(**variables)["prompt"]


if __name__ == "__main__":
    pass
