"""Routing and control logic for agent conversations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel
from typing_extensions import TypeVar


if TYPE_CHECKING:
    from llmling_agent.delegation import AgentPool


DecisionType = Literal["route", "talk_back", "end"]
TMessage = TypeVar("TMessage", default=str)
TCallback = TypeVar("TCallback")


class Decision(BaseModel):
    """Base class for all control decisions."""

    type: DecisionType
    reason: str


class RouteDecision(Decision):
    """Decision to forward to another agent without waiting."""

    type: Literal["route"] = "route"
    target_agent: str


class TalkBackDecision(Decision):
    """Decision to route and wait for response."""

    type: Literal["talk_back"] = "talk_back"
    target_agent: str


class EndDecision(Decision):
    """Decision to end conversation."""

    type: Literal["end"] = "end"


DecisionCallback = Callable[[TMessage, list[str]], Awaitable[Decision]]


class ConversationController[TMessage]:
    """Controls conversation flow through decisions."""

    def __init__(
        self,
        pool: AgentPool,
        decision_callback: DecisionCallback[TMessage],
    ):
        """Initialize controller.

        Args:
            pool: Agent pool containing available agents
            decision_callback: Function making routing decisions based on (structured)
                               messages
        """
        self.pool = pool
        self.decision_callback = decision_callback

    async def decide(self, message: TMessage) -> Decision:
        """Get decision for current conversation state.

        Args:
            message: Current (structured) message to make decision about

        Returns:
            Decision about how to proceed with conversation
        """
        return await self.decision_callback(message, self.pool.list_agents())


async def interactive_controller(message: str, available_agents: list[str]) -> Decision:
    """Interactive conversation control through console input."""
    print(f"\nMessage: {message}")
    print("\nWhat would you like to do?")
    print("1. Forward message (no wait)")
    print("2. Route and wait for response")
    print("3. End conversation")

    try:
        choice = int(input("> "))

        match choice:
            case 1 | 2:  # Route or TalkBack
                print("\nAvailable agents:")
                for i, name in enumerate(available_agents, 1):
                    print(f"{i}. {name}")
                agent_idx = int(input("Select agent: ")) - 1
                target = available_agents[agent_idx]
                reason = input("Reason: ")

                if choice == 1:
                    return RouteDecision(target_agent=target, reason=reason)
                return TalkBackDecision(target_agent=target, reason=reason)

            case 3:  # End
                reason = input("Reason for ending: ")
                return EndDecision(reason=reason)

            case _:
                return EndDecision(reason="Invalid choice")

    except (ValueError, IndexError):
        return EndDecision(reason="Invalid input")


async def rule_based_controller(message: str, available_agents: list[str]) -> Decision:
    """Route based on simple keyword matching."""
    msg = message.lower()

    # Define routing rules
    rules = [
        ("code", "code_expert", "Contains code-related query", True),
        ("data", "data_analyst", "Contains data-related query", True),
        ("test", "tester", "Related to testing", False),
        ("bug", "debugger", "Bug report or issue", True),
    ]

    # Check each rule
    for keyword, target, reason, needs_response in rules:
        if keyword in msg and target in available_agents:
            if needs_response:
                return TalkBackDecision(target_agent=target, reason=reason)
            return RouteDecision(target_agent=target, reason=reason)

    # Default route if no rules match
    default = "general_assistant"
    if default in available_agents:
        return TalkBackDecision(
            target_agent=default, reason="No specific routing rule matched"
        )

    # End if no suitable agent found
    return EndDecision(reason="No suitable agent available")


async def controlled_conversation(
    pool: AgentPool,
    initial_agent: str = "starter",
    initial_prompt: str = "Hello!",
    controller: ConversationController | None = None,
) -> None:
    """Run a controlled conversation between agents.

    Args:
        pool: Agent pool containing available agents
        initial_agent: Name of agent to start with
        initial_prompt: First message to start conversation
        controller: Optional controller (creates interactive one if None)
    """
    if controller is None:
        controller = ConversationController(pool, interactive_controller)

    current_agent = pool.get_agent(initial_agent)
    current_message = initial_prompt

    while True:
        # Get response from current agent
        response = await current_agent.run(current_message)
        decision = await controller.decide(response.content)

        match decision:
            case EndDecision():
                print(f"\nEnding conversation: {decision.reason}")
                break

            case RouteDecision():
                print(f"\nForwarding to {decision.target_agent}: {decision.reason}")
                next_agent = pool.get_agent(decision.target_agent)
                next_agent.outbox.emit(response, None)

            case TalkBackDecision():
                print(f"\nRouting to {decision.target_agent}: {decision.reason}")
                current_agent = pool.get_agent(decision.target_agent)
                current_message = response.content
