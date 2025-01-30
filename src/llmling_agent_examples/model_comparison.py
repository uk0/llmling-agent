"""Example of comparing different models using parallel teams."""

from __future__ import annotations

import asyncio
import tempfile

from llmling_agent.delegation import AgentPool


AGENT_CONFIG = """\
agents:
  overseer:
    name: "Model Comparison Coordinator"
    model: openai:gpt-4o-mini
    capabilities:
      can_add_teams: true
      can_list_teams: true
      can_delegate_tasks: true
      can_list_agents: true
      can_add_agents: true
    system_prompts:
      - You are a model comparison coordinator.
"""

PROMPT = """Please perform the following steps:

1. Create two agents:
   - Name: "gpt35_agent" using model "openai:gpt-3.5-turbo"
   - Name: "gpt4_agent" using model "openai:gpt-4o-mini"

2. Create a parallel team with the name "comparison_team".
  It should contain both agents just created.

3. Delegate this task to the team:
   "Explain the concept of quantum entanglement in exactly three sentences."

4. Analyze the responses, paying attention to:
   - Differences in explanation style and depth
   - Response time differences
   - Costs
   - Quality of the three-sentence constraint adherence
"""


async def run(config_path: str):
    """Run the model comparison example."""
    async with AgentPool[None](config_path) as pool:
        # Get the overseer agent with all needed capabilities
        overseer = pool.get_agent("overseer")
        print("\n=== Running Model Comparison ===")
        result = await overseer.run(PROMPT)
        print(result.content)


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(run(tmp.name))
