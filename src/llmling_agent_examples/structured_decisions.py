"""Smart Support Router Example."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from llmling_agent.delegation.controllers import CallbackConversationController
from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.router import (
    Decision,
    RouteDecision,
    TalkBackDecision,
)


class Category(str, Enum):
    TECH = "tech"
    BILLING = "billing"


class Ticket(BaseModel):
    """Minimal ticket with just enough to demonstrate routing."""

    category: Category
    urgent: bool


class Response(BaseModel):
    """Simple response type."""

    solved: bool


AGENT_CONFIG = """
agents:
  classifier:
    type: ai
    name: "Classifier"
    model: openai:gpt-4o-mini
    system_prompts:
      - Classify support requests into tech or billing tickets.

  tech_support:
    type: ai
    name: "Tech Support"
    model: openai:gpt-4o-mini
    system_prompts:
      - Handle technical support tickets.

  human:
    type: human
    name: "Human"
    system_prompts:
      - You handle urgent cases.
"""


async def smart_router(ticket: Ticket, pool: AgentPool) -> Decision:
    """Route based on category and urgency."""
    if ticket.urgent:
        # Can use pool features like team_task
        team = ["tech_support", "human"]
        await pool.team_task(str(ticket), team=team, mode="parallel")
        # Route based on team response
        return RouteDecision(target_agent="human", reason="Urgent case handled by team")

    # Simple routing for non-urgent cases
    agent_name = "tech_support" if ticket.category == Category.TECH else "billing"
    return TalkBackDecision(target_agent=agent_name, reason=f"{ticket.category} issue")


async def main():
    async with AgentPool.open(AGENT_CONFIG) as pool:
        # Get type-safe agents
        classifier = pool.get_agent("classifier", return_type=Ticket)

        # Create controller with pool-aware router
        controller = CallbackConversationController[Ticket](pool, smart_router)

        # Process request
        request = "My app is broken and I need urgent help!"
        ticket_msg = await classifier.run(request)
        decision = await controller.decide(ticket_msg.data)

        print(f"Ticket: {ticket_msg.data}")
        print(f"Decision: {decision}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
