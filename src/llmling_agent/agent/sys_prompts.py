from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from jinja2 import Environment
from toprompt import to_prompt


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.models.context import AgentContext


ToolInjectionMode = Literal["off", "all", "required"]
ToolUsageStyle = Literal["suggestive", "strict"]


DEFAULT_TEMPLATE = """\
{%- if inject_agent_info and agent.name %}You are {{ agent.name }}{% if agent.description %}. {{ agent.description }}{% endif %}.

{% endif -%}
{%- if inject_tools != "off" -%}
{%- set tools = agent.tools.get_tools("enabled") if inject_tools == "all" else agent.tools.get_required_tools() -%}
{%- if tools %}

{%- if tool_usage_style == "strict" %}
You MUST use these tools to complete your tasks:
{%- else %}
You have access to these tools:
{%- endif %}
{% for tool in tools %}
- {{ tool.name }}{% if tool.description %}: {{ tool.description }}{% endif %}{% if tool.requires_capability %} (requires {{ tool.requires_capability }}){% endif %}
{%- endfor %}

{%- if tool_usage_style == "strict" %}
Do not attempt to perform tasks without using appropriate tools.
{%- else %}
Use them when appropriate to complete your tasks.
{%- endif %}

{% endif -%}
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
        context: AgentContext | None = None,
        inject_agent_info: bool = True,
        inject_tools: ToolInjectionMode = "off",
        tool_usage_style: ToolUsageStyle = "suggestive",
    ):
        """Initialize prompt manager."""
        match prompts:
            case list():
                self.prompts = prompts
            case None:
                self.prompts = []
            case _:
                self.prompts = [prompts]
        self.context = context
        self.template = template
        self.dynamic = dynamic
        self.inject_agent_info = inject_agent_info
        self.inject_tools = inject_tools
        self.tool_usage_style = tool_usage_style
        self._cached = False
        self._env = Environment(enable_async=True)
        self._env.filters["to_prompt"] = to_prompt

    def __repr__(self) -> str:
        return (
            f"SystemPrompts(prompts={len(self.prompts)}, "
            f"dynamic={self.dynamic}, inject_agent_info={self.inject_agent_info}, "
            f"inject_tools={self.inject_tools!r})"
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | slice) -> AnyPromptType | list[AnyPromptType]:
        return self.prompts[idx]

    def add_library_prompt(self, identifier: str) -> None:
        """Add a prompt from the library by reference.

        Args:
            identifier: Name of the prompt in the library

        Raises:
            ValueError: If prompt not found or no context available
        """
        if not self.context:
            msg = "No context available to resolve library prompts"
            raise ValueError(msg)

        if not self.context.definition.prompts.system_prompts:
            msg = "No prompts available in library"
            raise ValueError(msg)

        prompt = self.context.definition.prompts.system_prompts.get(identifier)
        if not prompt:
            msg = f"Prompt {identifier!r} not found in library"
            raise ValueError(msg)

        self.prompts.append(prompt.content)

    async def refresh_cache(self) -> None:
        """Force re-evaluation of prompts."""
        evaluated = []
        for prompt in self.prompts:
            result = await to_prompt(prompt)
            evaluated.append(result)
        self.prompts = evaluated
        self._cached = True

    @asynccontextmanager
    async def temporary_prompt(
        self, prompt: AnyPromptType, exclusive: bool = False
    ) -> AsyncIterator[None]:
        """Temporarily override system prompts.

        Args:
            prompt: Single prompt or sequence of prompts to use temporarily
            exclusive: Whether to only use given prompt. If False, prompt will be
                       appended to the agents prompts temporarily.
        """
        original_prompts = self.prompts.copy()
        new_prompt = await to_prompt(prompt)
        self.prompts = [new_prompt] if not exclusive else [*self.prompts, new_prompt]
        try:
            yield
        finally:
            self.prompts = original_prompts

    async def format_system_prompt(self, agent: AnyAgent[Any, Any]) -> str:
        """Format complete system prompt."""
        if not self.dynamic and not self._cached:
            await self.refresh_cache()

        template = self._env.from_string(self.template or DEFAULT_TEMPLATE)
        result = await template.render_async(
            agent=agent,
            prompts=self.prompts,
            dynamic=self.dynamic,
            inject_agent_info=self.inject_agent_info,
            inject_tools=self.inject_tools,
            tool_usage_style=self.tool_usage_style,
        )
        return result.strip()
