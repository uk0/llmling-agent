# AI-Human Interaction

This example demonstrates how AI agents can interact with humans:

- Using agent capabilities for human interaction
- Setting up a human agent in the pool
- Allowing AI to request human input when needed

## Configuration

Our `human_agents.yml` sets up an AI assistant and a human agent:

```yaml
agents:
  assistant:
    model: openai:gpt-4o-mini
    capabilities:
      can_ask_agents: true  # Allow asking other agents (including humans)
    system_prompts:
      - |
        You are a helpful assistant. When you're not sure about something,
        don't hesitate to ask the human agent for guidance.

  human:
    type: "human"  # Special provider type for human interaction
    description: "A human who can provide answers"
```

## Implementation

Here's how we set up the human-AI interaction:

```python
from llmling_agent.delegation import AgentPool

# A question that requires human knowledge
QUESTION = """
What is the current status of Project DoomsDay?
This is crucial information that only a human would know.
If you don't know, ask the agent named "human".
"""

async def main():
    async with AgentPool[None]("human_agents.yml") as pool:
        assistant = pool.get_agent("assistant")
        # This will cause the assistant to interact with the human
        await assistant.run(QUESTION)
        # Show the complete conversation
        print(await assistant.conversation.format_history())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## How It Works

1. We set up two agents:
   - An AI assistant with `can_ask_agents` capability
   - A human agent using the special "human" provider type

2. When the AI assistant encounters a question it can't answer:
   - It recognizes the need for human input
   - Uses its `can_ask_agents` capability to interact with the human agent
   - Incorporates the human's response into its answer

3. The conversation might look like this:
   ```
   Assistant: I need to check about Project DoomsDay's status. Let me ask the human.
   Human: Project DoomsDay is currently in Phase 2, with 60% completion.
   Assistant: Based on the human's input, Project DoomsDay is in Phase 2 and is 60% complete.
   ```

This demonstrates how to:

- Enable AI-human collaboration
- Control when AI can request human input
- Integrate human knowledge into AI responses
