"""Builtin prompt provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent.models.prompts import SystemPrompt


class PromptReference:
    """Parsed prompt reference with optional version/variables."""

    def __init__(self, raw: str):
        # Parse provider:identifier@version[kwargs] syntax
        self.provider = "builtin"  # Default
        self.identifier = raw
        self.version = None
        self.variables = {}

        if ":" in raw:
            self.provider, rest = raw.split(":", 1)
            self.identifier = rest

        if "@" in self.identifier:
            self.identifier, version = self.identifier.split("@", 1)
            self.version = version

        if "[" in self.identifier:
            self.identifier, kwargs = self.identifier.split("[", 1)
            if not kwargs.endswith("]"):
                msg = "Invalid kwargs format"
                raise ValueError(msg)
            kwargs = kwargs[:-1]
            # Parse key=value pairs
            for pair in kwargs.split(","):
                k, v = pair.strip().split("=", 1)
                self.variables[k.strip()] = v.strip()


class BuiltinPromptProvider(BasePromptProvider):
    """Default provider using system prompts."""

    name = "builtin"

    def __init__(self, manifest_prompts: dict[str, SystemPrompt]):
        self.prompts = manifest_prompts

    async def get_prompt(
        self,
        identifier: str,
        version: str | None = None,  # Kept for interface compatibility
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get prompt content as string."""
        if identifier not in self.prompts:
            msg = f"Prompt not found: {identifier}"
            raise KeyError(msg)

        prompt = self.prompts[identifier]
        content = prompt.content

        if variables:
            # Format the content string directly
            return content.format(**variables)
        return content

    async def list_prompts(self) -> list[str]:
        """List available prompt identifiers."""
        return list(self.prompts.keys())
