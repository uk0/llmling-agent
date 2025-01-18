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
        experts = [
            pool.get_agent(name)
            for name in ["database_expert", "frontend_dev", "security_expert"]
        ]

        # Single expert selection
        pick = await coordinator.talk.pick(
            experts,
            task="Who should optimize our slow-running SQL queries?",
        )
        print("\nSingle expert selection:")
        # the result is type safe, pick.selection is an agent instance
        print(f"Selected: {pick.selection.name}")
        print(f"Because: {pick.reason}")

        # Multiple expert selection
        multi_pick = await coordinator.talk.pick_multiple(
            experts,
            task="Who should we assign to create a secure login page?",
            min_picks=2,
        )
        print("\nTeam selection:")
        # also here type-safe result
        print("Selected:", ", ".join(e.name for e in multi_pick.selections))
        print(f"Because: {multi_pick.reason}")


if __name__ == "__main__":
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
