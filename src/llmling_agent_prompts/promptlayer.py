"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from promptlayer import PromptLayer

from llmling_agent.prompts.models import PromptTemplate
from llmling_agent_prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent.models.prompt_hubs import PromptLayerConfig


class PromptLayerProvider(BasePromptProvider):
    name = "promptlayer"
    supports_versions = True

    def __init__(self, config: PromptLayerConfig):
        self.client = PromptLayer(api_key=config.api_key)

    async def get_prompt(
        self,
        prompt_id: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        # PromptLayer primarily tracks prompts used with their API
        # But also allows template management
        template = self.client.run(prompt_id)
        return PromptTemplate.parse_obj(template)
