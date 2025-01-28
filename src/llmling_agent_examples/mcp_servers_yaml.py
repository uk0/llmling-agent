"""Example: Two agents working together to explore git commit history using pool."""

from __future__ import annotations

from llmling_agent.delegation import AgentPool


AGENT_CONFIG = """\
mcp_servers:
  - "uvx mcp-server-git"

agents:
  picker:
    model: openai:gpt-4o-mini
    description: Git commit history explorer
    system_prompts:
      - You are a specialist in looking up git commits using your tools from the current working directory.
    connections:
      - type: node
        name: analyzer

  analyzer:
    model: openai:gpt-4o-mini
    description: Git commit analyzer
    system_prompts:
      - You are an expert in retrieving and returning information about a specific commit from the current working directory.
"""  # noqa: E501


async def run(config_path: str):
    async with AgentPool[None](config_path) as pool:
        # Get agents (connections already set up from YAML)
        picker = pool.get_agent("picker")
        analyzer = pool.get_agent("analyzer")

        # Register handlers to see messages
        picker.message_sent.connect(lambda msg: print(msg.format()))
        analyzer.message_sent.connect(lambda msg: print(msg.format()))

        # Start the chain
        await picker.run("Get the latest commit hash!")


if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        import asyncio

        asyncio.run(run(tmp.name))
