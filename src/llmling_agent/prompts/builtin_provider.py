"""Builtin prompt provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2 import Template, meta

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent.models.prompts import SystemPrompt


class BuiltinPromptProvider(BasePromptProvider):
    """Default provider using system prompts."""

    supports_variables = True
    name = "builtin"

    def __init__(self, manifest_prompts: dict[str, SystemPrompt]):
        from jinja2 import Environment

        from llmling_agent_functional.run import run_agent, run_agent_sync

        self.prompts = manifest_prompts
        self.env = Environment(autoescape=False, enable_async=True)
        self.env.globals["run_agent"] = run_agent
        self.env.globals["run_agent_sync"] = run_agent_sync

    async def get_prompt(
        self,
        identifier: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get prompt content as string."""
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
                msg = (
                    f"Prompt {identifier} requires variables: {', '.join(required_vars)}"
                )
                raise KeyError(msg)

            # Check for missing required variables
            missing_vars = required_vars - set(variables or {})
            if missing_vars:
                _vars = ", ".join(missing_vars)
                msg = f"Missing required variables for prompt {identifier}: {_vars}"
                raise KeyError(msg)

            try:
                template = Template(content)
                content = template.render(**variables)
            except Exception as e:
                msg = f"Failed to render prompt {identifier}: {e}"
                raise ValueError(msg) from e

        return content

    async def list_prompts(self) -> list[str]:
        """List available prompt identifiers."""
        return list(self.prompts.keys())
