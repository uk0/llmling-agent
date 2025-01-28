from typing import Any

import openlit

from llmling_agent.models.prompt_hubs import OpenLITConfig
from llmling_agent.prompts.models import PromptTemplate


class OpenLITProvider:
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
        meta_properties: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        """Get prompt from OpenLIT.

        Args:
            name: Name to fetch a unique prompt
            version: Optional version string
            variables: Optional variables for prompt compilation
            meta_properties: Optional meta-properties for access history
        """
        try:
            result = openlit.get_prompt(
                url=self._config.url,  # uses OPENLIT_URL env var if not set
                api_key=self._config.api_key,  # uses OPENLIT_API_KEY env var if not set
                name=name,
                version=version,
                should_compile=bool(variables),  # Only compile if variables provided
                variables=variables or {},
                meta_properties=meta_properties or {},
            )

            from llmling_agent.prompts.models import PromptTemplate

            assert result
            return PromptTemplate.model_validate(result["res"])

        except Exception as e:
            msg = f"Failed to load prompt using name={name}: {e}"
            raise RuntimeError(msg) from e
