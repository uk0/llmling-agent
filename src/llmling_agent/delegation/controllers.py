"""Controller implementations for agent conversations."""

from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.agent import Agent
from llmling_agent.delegation.callbacks import DecisionCallback
from llmling_agent.delegation.router import (
    AwaitResponseDecision,
    Decision,
    EndDecision,
    RouteDecision,
    RoutingConfig,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.delegation.pool import AgentPool


logger = get_logger(__name__)


class ConversationController[TMessage](ABC):
    """Base class for conversation controllers."""

    @abstractmethod
    async def decide(self, message: TMessage) -> Decision:
        """Make routing decision based on message."""


class CallbackConversationController[TMessage](ConversationController[TMessage]):
    """Controller that uses a callback function for decisions."""

    def __init__(
        self,
        pool: "AgentPool",
        decision_callback: DecisionCallback[TMessage],
    ):
        self.pool = pool
        self.decision_callback = decision_callback

    async def decide(self, message: TMessage) -> Decision:
        """Execute callback and handle sync/async appropriately."""
        result = self.decision_callback(message, self.pool)

        # Check if result is awaitable and await if needed
        if inspect.isawaitable(result):
            return await result
        return result


class RuleBasedController(ConversationController[str]):
    """Controller using predefined routing rules."""

    def __init__(
        self,
        pool: "AgentPool",
        config: RoutingConfig,
    ):
        self.pool = pool
        self.config = config

    async def decide(self, message: str) -> Decision:
        msg = message if self.config.case_sensitive else message.lower()

        # Check each rule in priority order
        for rule in sorted(self.config.rules, key=lambda r: r.priority):
            keyword = rule.keyword if self.config.case_sensitive else rule.keyword.lower()

            if keyword not in msg:
                continue

            # Skip if target doesn't exist
            if rule.target not in self.pool.list_agents():
                logger.debug(
                    "Target agent %s not available for rule: %s",
                    rule.target,
                    rule.keyword,
                )
                continue

            # Skip if capability required but not available
            if rule.requires_capability:
                agent = self.pool.get_agent(rule.target)
                if not agent._context.capabilities.has_capability(
                    rule.requires_capability
                ):
                    logger.debug(
                        "Agent %s missing required capability: %s",
                        rule.target,
                        rule.requires_capability,
                    )
                    continue

            # Create appropriate decision type
            if rule.wait_for_response:
                return AwaitResponseDecision(target_agent=rule.target, reason=rule.reason)
            return RouteDecision(target_agent=rule.target, reason=rule.reason)

        # Use default route if configured
        if self.config.default_target:
            return AwaitResponseDecision(
                target_agent=self.config.default_target, reason=self.config.default_reason
            )

        # End if no route found
        return EndDecision(reason="No matching rule or default route")


async def interactive_controller(message: str, pool: "AgentPool") -> Decision:
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
    controller = CallbackConversationController(pool, decision_callback)

    # Handle both string and Agent
    current_agent = (
        initial_agent
        if isinstance(initial_agent, Agent)
        else pool.get_agent(initial_agent)
    )
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

            case AwaitResponseDecision():
                print(f"\nRouting to {decision.target_agent}: {decision.reason}")
                current_agent = pool.get_agent(decision.target_agent)
                current_message = response.content
