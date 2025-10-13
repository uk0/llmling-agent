"""OpenLIT prompt provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import openlit  # pyright: ignore

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent_config.prompt_hubs import OpenLITConfig


class OpenLITProvider(BasePromptProvider):
    """Provider for prompts managed in OpenLIT."""

    name = "openlit"
    supports_versions = True
    supports_variables = True

    def __init__(self, config: OpenLITConfig):
        """Initialize OpenLIT provider."""
        self._config = config

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get prompt from OpenLIT.

        Args:
            name: Name to fetch a unique prompt
            version: Optional version string
            variables: Optional variables for prompt compilation
        """
        try:
            result = openlit.get_prompt(
                url=self._config.url,  # uses OPENLIT_URL env var if not set
                api_key=self._config.api_key,  # uses OPENLIT_API_KEY env var if not set
                name=name,
                version=version,
                should_compile=bool(variables),  # Only compile if variables provided
                variables=variables or {},
                meta_properties={},
            )

            assert result
            return result["res"]

        except Exception as e:
            msg = f"Failed to load prompt using name={name}: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    from llmling_agent_config.prompt_hubs import OpenLITConfig

    config = OpenLITConfig()
    prompt_hub = OpenLITProvider(config)

    async def main():
        print(await prompt_hub.get_prompt("test-b4d4"))

    asyncio.run(main())
