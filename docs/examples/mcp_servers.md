# MCP Server Integration Example

This example demonstrates how to use MCP (Model Control Protocol) servers with agents to explore git commit history. It shows:
- Setting up agents with MCP servers
- Communication between agents using MCP tools
- Handling of commit information in a structured way

## Implementation

Here's how we set up the commit analysis workflow:

```python
from llmling_agent import Agent

# Create agents with MCP server access
picker = Agent(
    name="picker",
    model="openai:gpt-4o-mini",
    system_prompt="You are a specialist in looking up git commits using your tools from the current working directory.",
    mcp_servers=["uvx mcp-server-git"]
)

analyzer = Agent(
    name="analyzer",
    model="openai:gpt-4o-mini",
    system_prompt="You are an expert in retrieving and returning information about a specific commit from the current working directoy.",
    mcp_servers=["uvx mcp-server-git"]
)

# Connect picker to analyzer
picker >> analyzer

# Register message handlers to see the messages
picker.message_sent.connect(lambda msg: print(msg.format()))
analyzer.message_sent.connect(lambda msg: print(msg.format()))

async def main():
    # For MCP servers, we need async context
    async with picker, analyzer:
        # Start the chain by asking picker for the latest commit
        await picker.run("Get the latest commit hash!")

if __name__ == "__main__":
    import anyio
    anyio.run(main)
```

## How It Works

1. Two agents are set up with access to a Git MCP server:

- `picker`: Specialized in finding commits
- `analyzer`: Specialized in analyzing commit details

2. The agents are connected using `>>` operator, so the picker's output flows to the analyzer

3. Message handlers are registered to display the conversation

4. When run, the workflow:

- Picker finds the latest commit hash
- Automatically forwards to analyzer
- Analyzer provides detailed commit information

Example Output:
```
CommitPicker: The latest commit hash is **9bcd7718dbc33f16239d0522ca677ed75bac997b**.

CommitAnalyzer: The latest commit with hash **9bcd7718dbc33f16239d0522ca677ed75bac997b**
includes the following details:

- **Author:** Philipp Temminghoff
- **Date:** January 20, 2024, at 01:59:43 (local time)
- **Commit Message:** chore: docs

### Changes made:
...
```

This demonstrates:

- MCP server integration for Git operations
- Agent chaining with automatic message forwarding
- Structured commit analysis workflow
- Async contexts for proper server handling
