from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.models.prompt_hubs import (
    LangfuseConfig,
    OpenLITConfig,
    PromptLayerConfig,
)
from llmling_agent.prompts.builtin_provider import BuiltinPromptProvider
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from llmling_agent.models.prompts import PromptConfig
    from llmling_agent.prompts.base import BasePromptProvider

logger = get_logger(__name__)


def parse_prompt_reference(reference: str) -> tuple[str, str, str | None, dict[str, str]]:
    """Parse a prompt reference string.

    Args:
        reference: Format [provider:]name[@version][?var1=val1,var2=val2]

    Returns:
        Tuple of (provider, identifier, version, variables)
        Provider defaults to "builtin" if not specified
    """
    provider = "builtin"
    version = None
    variables = {}

    # Split provider and rest
    if ":" in reference:
        provider, identifier = reference.split(":", 1)
    else:
        identifier = reference

    # Handle version
    if "@" in identifier:
        identifier, version = identifier.split("@", 1)

    # Handle query parameters
    if "?" in identifier:
        identifier, query = identifier.split("?", 1)
        for pair in query.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            variables[key.strip()] = value.strip()

    return provider, identifier.strip(), version, variables


class PromptManager(TaskManagerMixin):
    """Manages multiple prompt providers.

    Handles:
    - Provider initialization and cleanup
    - Prompt reference parsing and resolution
    - Access to prompts from different sources
    """

    def __init__(self, config: PromptConfig):
        """Initialize prompt manager.

        Args:
            config: Prompt configuration including providers
        """
        super().__init__()
        self.config = config
        self.providers: dict[str, BasePromptProvider] = {}

        # Always register builtin provider
        self.providers["builtin"] = BuiltinPromptProvider(config.system_prompts)

        # Register configured providers
        if config.providers:
            for provider_config in config.providers:
                match provider_config:
                    case LangfuseConfig():
                        from llmling_agent_prompts.langfuse_hub import LangfusePromptHub

                        self.providers["langfuse"] = LangfusePromptHub(provider_config)
                    case OpenLITConfig():
                        from llmling_agent_prompts.openlit import OpenLITProvider

                        self.providers["openlit"] = OpenLITProvider(provider_config)
                    case PromptLayerConfig():
                        from llmling_agent_prompts.promptlayer import PromptLayerProvider

                        self.providers["promptlayer"] = PromptLayerProvider(
                            provider_config
                        )

    async def get(self, reference: str) -> str:
        """Get a prompt using identifier syntax.

        Args:
            reference: Prompt identifier in format:
                      [provider:]name[@version][?var1=val1,var2=val2]

        Examples:
            await prompts.get("error_handler")  # Builtin prompt
            await prompts.get("langfuse:code_review@v2?style=detailed")
            await prompts.get("openlit:explain?context=web,detail=high")
        """
        provider_name, identifier, version, variables = parse_prompt_reference(reference)

        if provider_name not in self.providers:
            msg = f"Unknown prompt provider: {provider_name}"
            raise KeyError(msg)

        provider = self.providers[provider_name]

        try:
            # Only pass version/variables if provider supports them
            kwargs: dict[str, Any] = {}
            if provider.supports_versions and version:
                kwargs["version"] = version
            if provider.supports_variables and variables:
                kwargs["variables"] = variables

            return await provider.get_prompt(identifier, **kwargs)

        except Exception as e:
            msg = f"Failed to get prompt {reference!r} from {provider_name}"
            raise RuntimeError(msg) from e

    def get_sync(self, reference: str) -> str:
        """Synchronous wrapper for get().

        Note: This is a temporary solution for initialization contexts.
        Should be used only when async operation is not possible.
        """
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.get(reference))
        except RuntimeError:
            # No running loop - create new one
            return asyncio.run(self.get(reference))

    async def list_prompts(self, provider: str | None = None) -> dict[str, list[str]]:
        """List available prompts.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Dict mapping provider names to their available prompts
        """
        result = {}
        providers = {provider: self.providers[provider]} if provider else self.providers

        for name, p in providers.items():
            try:
                prompts = await p.list_prompts()
                result[name] = prompts
            except Exception:
                logger.exception("Failed to list prompts from %s", name)
                continue

        return result

    def cleanup(self):
        """Clean up providers."""
        self.providers.clear()

    def get_default_provider(self) -> str:
        """Get default provider name.

        Returns configured default or "builtin"
        """
        return "builtin"
