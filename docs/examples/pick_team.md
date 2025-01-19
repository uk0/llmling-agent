# Expert Selection with pick() and pick_multiple()

This example demonstrates LLMling-agent's type-safe selection methods:

- Using pick() for single expert selection
- Using pick_multiple() for team selection
- Type-safe results with reasoning
- Team-based agent selection

## Configuration

Our `experts.yml` defines a coordinator and three specialists:

```yaml
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
```

## Implementation

Here's how we use the selection methods:

```python
from llmling_agent import AgentPool

async def main():
    async with AgentPool[None]("experts.yml") as pool:
        coordinator = pool.get_agent("coordinator")

        # Create team of available experts
        experts = pool.create_team([
            "database_expert",
            "frontend_dev",
            "security_expert"
        ])

        # Single expert selection
        task = "Who should optimize our slow-running SQL queries?"
        pick = await coordinator.talk.pick(experts, task=task)
        # Type-safe result: pick.selection is an Agent instance
        assert pick.selection in experts
        print(f"Selected: {pick.selection.name} Reason: {pick.reason}")

        # Multiple expert selection
        task = "Who should we assign to create a secure login page?"
        multi_pick = await coordinator.talk.pick_multiple(
            experts,
            task=task,
            min_picks=2
        )
        # Also type-safe: selections is list[Agent]
        selected = ", ".join(e.name for e in multi_pick.selections)
        print(f"Selected: {selected} Reason: {multi_pick.reason}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## How It Works

1. Single Selection (pick):
   - Takes a team of agents and a task description
   - Returns a single expert with reasoning
   - Result is type-safe: `Pick[Agent]`

2. Multiple Selection (pick_multiple):
   - Takes same inputs plus min/max picks
   - Returns multiple experts with reasoning
   - Result is type-safe: `MultiPick[Agent]`

Example Output:
```
Selected: database_expert
Reason: The task specifically involves SQL query optimization, which is the database expert's primary specialty.

Selected: frontend_dev, security_expert
Reason: Creating a secure login page requires both frontend expertise for the user interface and security expertise for proper authentication implementation.
```

This demonstrates:

- Type-safe agent selection
- Reasoned decision-making
- Team-based operations
- Flexible expert allocation
