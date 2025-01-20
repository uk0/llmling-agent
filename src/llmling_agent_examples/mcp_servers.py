"""Example: Two agents working together to explore git commit history."""

from __future__ import annotations

from llmling_agent import Agent


PICKER = """
You are a specialist in looking up git commits using your tools
from the current working directory."
"""
ANALYZER = """
You are an expert in retrieving and returning information
about a specific commit from the current working directoy."
"""

MODEL = "openai:gpt-4o-mini"
SERVERS = ["uvx mcp-server-git"]

picker = Agent[None](model=MODEL, system_prompt=PICKER, mcp_servers=SERVERS)
analyzer = Agent[None](model=MODEL, system_prompt=ANALYZER, mcp_servers=SERVERS)

# Connect picker to analyzer
picker >> analyzer

# Register message handlers to see the messages
picker.message_sent.connect(lambda msg: print(msg.format()))
analyzer.message_sent.connect(lambda msg: print(msg.format()))


async def main():
    # For MCP servers, we need async context.
    async with picker, analyzer:
        # Start the chain by asking picker for the latest commit
        await picker.run("Get the latest commit hash! ")


if __name__ == "__main__":
    import anyio

    anyio.run(main)


"""
Output:

CommitPicker: The latest commit hash is **9bcd7718dbc33f16239d0522ca677ed75bac997b**.
CommitAnalyzer: The latest commit with hash **9bcd7718dbc33f16239d0522ca677ed75bac997b**
includes the following details:

- **Author:** Philipp Temminghoff
- **Date:** January 20, 2025, at 01:59:43 (local time)
- **Commit Message:** chore: docs

### Changes made:
...
"""
