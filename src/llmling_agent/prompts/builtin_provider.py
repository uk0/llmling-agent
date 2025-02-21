"""Builtin prompt provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling.config.runtime import RuntimeConfig

    from llmling_agent_config.prompts import SystemPrompt


class BuiltinPromptProvider(BasePromptProvider):
    """Default provider using system prompts."""

    supports_variables = True
    name = "builtin"

    def __init__(self, manifest_prompts: dict[str, SystemPrompt]):
        from jinjarope import Environment

        self.prompts = manifest_prompts
        self.env = Environment(autoescape=False, enable_async=True)

    async def get_prompt(
        self,
        identifier: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get prompt content as string."""
        from jinja2 import Template, meta

        if identifier not in self.prompts:
            msg = f"Prompt not found: {identifier}"
            raise KeyError(msg)

        prompt = self.prompts[identifier]
        content = prompt.content

        # Parse template to find required variables
        ast = self.env.parse(content)
        required_vars = meta.find_undeclared_variables(ast)

        if variables:
            # Check for unknown variables
            unknown_vars = set(variables) - required_vars
            if unknown_vars:
                _vars = ", ".join(unknown_vars)
                msg = f"Unknown variables for prompt {identifier}: {_vars}"
                raise KeyError(msg)

        if required_vars:
            if not variables:
                req = ", ".join(required_vars)
                msg = f"Prompt {identifier} requires variables: {req}"
                raise KeyError(msg)

            # Check for missing required variables
            missing_vars = required_vars - set(variables or {})
            if missing_vars:
                _vars = ", ".join(missing_vars)
                msg = f"Missing required variables for prompt {identifier}: {_vars}"
                raise KeyError(msg)

            try:
                template = Template(content, enable_async=True)
                content = await template.render_async(**variables)
            except Exception as e:
                msg = f"Failed to render prompt {identifier}: {e}"
                raise ValueError(msg) from e

        return content

    async def list_prompts(self) -> list[str]:
        """List available prompt identifiers."""
        return list(self.prompts.keys())


class RuntimePromptProvider(BasePromptProvider):
    """Provider for prompts from RuntimeConfig."""

    name = "runtime"
    supports_variables = True

    def __init__(self, runtime_config: RuntimeConfig):
        self.runtime = runtime_config

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        prompt = self.runtime.get_prompt(name)
        if variables:
            messages = await prompt.format(variables)
        messages = await prompt.format()
        return "\n".join(m.get_text_content() for m in messages)

    async def list_prompts(self) -> list[str]:
        return [p.name for p in self.runtime.get_prompts() if p.name]
