"""Smart Support Router Example."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.router import (
    AgentRouter,
    AwaitResponseDecision,
    CallbackRouter,
    Decision,
    RouteDecision,
)


class Ticket(BaseModel):
    """Minimal ticket with just enough to demonstrate routing."""

    category: Literal["tech", "billing"]
    urgent: bool


AGENT_CONFIG = """
agents:
  classifier:
    type: pydantic_ai
    name: "Classifier"
    model: openai:gpt-4o-mini
    system_prompts:
      - Classify support requests into tech or billing tickets.

  tech_support:
    type: pydantic_ai
    name: "Tech Support"
    model: openai:gpt-4o-mini
    system_prompts:
      - Handle technical support tickets.

  billing_support:
    type: pydantic_ai
    name: "Billing Support"
    model: openai:gpt-4o-mini
    system_prompts:
      - Handle billing and payment issues.
"""


async def smart_router(ticket: Ticket, pool: AgentPool, router: AgentRouter) -> Decision:
    """Route based on category and urgency."""
    agent = "tech_support" if ticket.category == "tech" else "billing_support"
    if ticket.urgent:
        # For urgent cases, route directly to appropriate specialist
        reason = f"Urgent {ticket.category} issue needs immediate attention"
        return AwaitResponseDecision(target_agent=agent, reason=reason)

    # Non-urgent cases can be handled asynchronously
    return RouteDecision(target_agent=agent, reason=f"Standard {ticket.category} issue")


async def main(config_path: str):
    async with AgentPool[None](config_path) as pool:
        # Get type-safe classifier agent
        classifier = pool.get_agent("classifier", return_type=Ticket)
        # Create controller with pool-aware router
        controller = CallbackRouter[Ticket](pool, smart_router)
        request = "My app is broken and I need urgent help!"
        ticket_msg = await classifier.run(request)
        decision = await controller.decide(ticket_msg.data)

        print(f"Ticket: {ticket_msg.data}")
        print(f"Decision: {decision}")


if __name__ == "__main__":
    import asyncio
    import logging
    import tempfile

    logging.basicConfig(level=logging.DEBUG)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
