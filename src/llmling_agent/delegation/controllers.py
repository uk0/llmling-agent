"""Controller implementations for agent conversations."""

from typing import TYPE_CHECKING, Any

from llmling_agent.agent import Agent
from llmling_agent.delegation.callbacks import DecisionCallback
from llmling_agent.delegation.router import (
    AgentRouter,
    AwaitResponseDecision,
    CallbackRouter,
    Decision,
    EndDecision,
    RouteDecision,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.delegation.pool import AgentPool


logger = get_logger(__name__)


async def interactive_controller(
    message: str, pool: "AgentPool", agent_router: AgentRouter
) -> Decision:
    """Interactive conversation control through console input."""
    print(f"\nMessage: {message}")
    print("\nWhat would you like to do?")
    print("1. Forward message (no wait)")
    print("2. Route and wait for response")
    print("3. End conversation")

    try:
        match choice := int(input("> ")):
            case 1 | 2:  # Route or TalkBack
                print("\nAvailable agents:")
                # Use pool's list_agents instead of passed list
                agents = pool.list_agents()
                for i, name in enumerate(agents, 1):
                    agent = pool.get_agent(name)
                    print(f"{i}. {name} ({agent.description or 'No description'})")

                agent_idx = int(input("Select agent: ")) - 1
                target = agents[agent_idx]
                reason = input("Reason: ")

                if choice == 1:
                    return RouteDecision(target_agent=target, reason=reason)
                return AwaitResponseDecision(target_agent=target, reason=reason)

            case 3:  # End
                reason = input("Reason for ending: ")
                return EndDecision(reason=reason)

            case _:
                return EndDecision(reason="Invalid choice")

    except (ValueError, IndexError):
        return EndDecision(reason="Invalid input")


async def controlled_conversation(
    pool: "AgentPool",
    initial_agent: str | Agent[Any] = "starter",
    initial_prompt: str = "Hello!",
    decision_callback: DecisionCallback = interactive_controller,
) -> None:
    """Run a controlled conversation between agents.

    Args:
        pool: Agent pool containing available agents
        initial_agent: Name of agent to start with
        initial_prompt: First message to start conversation
        decision_callback: Optional decision callback (defaults to interactive)
    """
    controller = CallbackRouter(pool, decision_callback)

    agent = (
        initial_agent
        if isinstance(initial_agent, Agent)
        else pool.get_agent(initial_agent)
    )
    current_message = initial_prompt

    while True:
        # Get response from current agent
        response = await agent.run(current_message)
        decision = await controller.decide(response.content)

        # Execute the decision with the response message
        await decision.execute(response, agent, pool)

        match decision:
            case EndDecision():
                break
            case RouteDecision():
                # Message already forwarded in execute(), continue loop
                continue
            case AwaitResponseDecision():
                agent = pool.get_agent(decision.target_agent)
                current_message = str(response.content)
