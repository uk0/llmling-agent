"""Example: Using pick() and pick_multiple() for expert selection."""

from __future__ import annotations

from llmling_agent import AgentPool


AGENT_CONFIG = """
agents:
  coordinator:
    model: openai:gpt-4o-mini
    system_prompts:
      - You select the most suitable expert(s) for each task.

  database_expert:
    model: openai:gpt-4o-mini
    description: Expert in SQL optimization and database design.

  frontend_dev:
    model: openai:gpt-4o-mini
    description: Specialist in React and modern web interfaces.

  security_expert:
    model: openai:gpt-4o-mini
    description: Expert in penetration testing and security audits.
"""


async def main(config_path: str):
    async with AgentPool[None](config_path) as pool:
        coordinator = pool.get_agent("coordinator")
        experts = pool.create_team(["database_expert", "frontend_dev", "security_expert"])

        # Single expert selection
        task = "Who should optimize our slow-running SQL queries?"
        pick = await coordinator.talk.pick(experts, task=task)
        # the result is type safe, pick.selection is an agent instance
        assert pick.selection in experts
        print(f"Selected: {pick.selection.name} Reason: {pick.reason}")

        # Multiple expert selection
        task = "Who should we assign to create a secure login page?"
        multi_pick = await coordinator.talk.pick_multiple(experts, task=task, min_picks=2)
        # also here type-safe result
        selected = ", ".join(e.name for e in multi_pick.selections)
        print(f"Selected: {selected} Reason: {multi_pick.reason}")


if __name__ == "__main__":
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
