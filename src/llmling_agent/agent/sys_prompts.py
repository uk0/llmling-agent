from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2 import Environment
from toprompt import to_prompt


if TYPE_CHECKING:
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent


DEFAULT_TEMPLATE = """\
{%- if inject_agent_info and agent.name %}You are {{ agent.name }}{% if agent.description %}. {{ agent.description }}{% endif %}.

{% endif -%}
{%- for prompt in prompts %}
{{ prompt|to_prompt if dynamic else prompt }}
{%- if not loop.last %}

{% endif %}
{%- endfor %}"""  # noqa: E501


class SystemPrompts:
    """Manages system prompts for an agent."""

    def __init__(
        self,
        prompts: AnyPromptType | list[AnyPromptType] | None = None,
        template: str | None = None,
        dynamic: bool = True,
        inject_agent_info: bool = True,
    ):
        """Initialize prompt manager."""
        match prompts:
            case list():
                self.prompts = prompts
            case None:
                self.prompts = []
            case _:
                self.prompts = [prompts]
        self.template = template
        self.dynamic = dynamic
        self.inject_agent_info = inject_agent_info
        self._cached = False
        self._env = Environment(enable_async=True)
        self._env.filters["to_prompt"] = to_prompt

    def __repr__(self) -> str:
        return (
            f"SystemPrompts(prompts={len(self.prompts)}, "
            f"dynamic={self.dynamic}, inject_agent_info={self.inject_agent_info})"
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | slice) -> AnyPromptType | list[AnyPromptType]:
        return self.prompts[idx]

    async def refresh_cache(self) -> None:
        """Force re-evaluation of prompts."""
        evaluated = []
        for prompt in self.prompts:
            result = await to_prompt(prompt)
            evaluated.append(result)
        self.prompts = evaluated
        self._cached = True

    async def format_system_prompt(self, agent: AnyAgent[Any, Any]) -> str:
        """Format complete system prompt."""
        if not self.dynamic and not self._cached:
            await self.refresh_cache()

        template = self._env.from_string(self.template or DEFAULT_TEMPLATE)
        return await template.render_async(
            agent=agent,
            prompts=self.prompts,
            dynamic=self.dynamic,
            inject_agent_info=self.inject_agent_info,
        )
