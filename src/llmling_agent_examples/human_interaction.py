"""Example of AI-Human interaction using agent capabilities.

This example shows how an AI agent can ask for human input when needed,
using the can_ask_agents capability and a human agent in the pool.
"""

from __future__ import annotations

from llmling_agent.delegation import AgentPool


AGENT_CONFIG = """
agents:
  assistant:
    model: openai:gpt-4o-mini
    capabilities:
      can_ask_agents: true
    system_prompts:
      - |
        You are a helpful assistant. When you're not sure about something,
        don't hesitate to ask the human agent for guidance.

  human:
    type: "human"
    description: "A human who can provide answers"
"""

QUESTION = """
What is the current status of Project DoomsDay?
This is crucial information that only a human would know.
If you don't know, ask the agent named "human".
"""


async def main(config_path: str):
    async with AgentPool[None](config_path) as pool:
        assistant = pool.get_agent("assistant")
        await assistant.run(QUESTION)
        print(await assistant.conversation.format_history())


if __name__ == "__main__":
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
