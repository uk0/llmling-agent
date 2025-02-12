import os

import pytest

from llmling_agent import AgentsManifest
from llmling_agent.observability import registry


MANIFEST = """
observability:
  enabled: true
  providers:
    - type: logfire
      token: {logfire_token}
      service_name: integration_test
    - type: agentops
      api_key: {agentops_key}
      tags: ["test", "integration"]
    - type: langsmith
      api_key: {langsmith_key}
      project_name: integration_test
    - type: arize
      api_key: {arize_key}
      space_key: integration_test

agents:
  test_agent:
    name: test_agent
    provider:
      type: pydantic_ai
      model:
        type: test
        custom_result_text: "Test response"
    system_prompts:
      - You are a helpful assistant.
"""

# Skip if no providers configured
requires_providers = pytest.mark.skipif(
    not any([
        os.getenv("LOGFIRE_TOKEN"),
        os.getenv("AGENTOPS_API_KEY"),
        os.getenv("LANGSMITH_API_KEY"),
        os.getenv("ARIZE_API_KEY"),
    ]),
    reason="No observability providers configured",
)


@requires_providers
@pytest.mark.asyncio
async def test_provider_integration():
    """Test that all configured providers can be initialized and work together."""
    manifest_str = MANIFEST.format(
        logfire_token=os.getenv("LOGFIRE_TOKEN", "dummy_token"),
        agentops_key=os.getenv("AGENTOPS_API_KEY", "dummy_key"),
        langsmith_key=os.getenv("LANGSMITH_API_KEY", "dummy_key"),
        arize_key=os.getenv("ARIZE_API_KEY", "dummy_key"),
    )

    manifest = AgentsManifest.from_yaml(manifest_str)

    async with manifest.pool as pool:
        agent = pool.get_agent("test_agent")
        # Run a simple prompt
        result = await agent.run("Hello!")
        # We don't assert anything specific - just verify no errors occurred
        assert result.content == "Test response"
        assert len(registry.providers) >= 4  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
