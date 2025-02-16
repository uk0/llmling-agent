"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llmling_agent_config.prompt_hubs import PromptHubConfig  # noqa: TC001


SystemPromptCategory = Literal["role", "methodology", "quality", "task"]

DEFAULT_TEMPLATE = """
{%- for prompt in role_prompts %}
Role: {{ prompt.content }}
{%- endfor %}

{%- for prompt in methodology_prompts %}
Method: {{ prompt.content }}
{%- endfor %}

{%- for prompt in quality_prompts %}
Quality Check: {{ prompt.content }}
{%- endfor %}

{%- for prompt in task_prompts %}
Task: {{ prompt.content }}
{%- endfor %}
"""


class SystemPrompt(BaseModel):
    """Individual system prompt definition."""

    content: str
    """The actual prompt text."""

    type: SystemPromptCategory = "role"
    """Categorization for template organization."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class PromptConfig(BaseModel):
    """Complete prompt configuration."""

    system_prompts: dict[str, SystemPrompt] = Field(default_factory=dict)
    """Mapping of system prompt identifiers to their definitions."""

    template: str | None = None
    """Optional template for combining prompts.
    Has access to prompts grouped by type."""

    providers: list[PromptHubConfig] = Field(default_factory=list)
    """List of external prompt providers to use."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")

    def format_prompts(self, identifiers: list[str] | None = None) -> str:
        """Format selected prompts using template.

        Args:
            identifiers: Optional list of prompt IDs to include.
                       If None, includes all prompts.
        """
        # Filter prompts if identifiers provided
        prompts = (
            {k: v for k, v in self.system_prompts.items() if k in identifiers}
            if identifiers
            else self.system_prompts
        )

        # Group prompts by type for template
        by_type = {
            "role_prompts": [p for p in prompts.values() if p.type == "role"],
            "methodology_prompts": [
                p for p in prompts.values() if p.type == "methodology"
            ],
            "quality_prompts": [p for p in prompts.values() if p.type == "quality"],
            "task_prompts": [p for p in prompts.values() if p.type == "task"],
        }

        # Render template
        from jinja2 import Template

        template = Template(self.template or DEFAULT_TEMPLATE)
        return template.render(**by_type)
