"""Smart Support Router Example.

This example demonstrates:
1. Type-safe agent communication using StructuredAgent
2. Generic routing with structured messages
3. Mix of sync and async decision makers
4. Different response types for different agents
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from llmling_agent.delegation import (
    AgentPool,
    AwaitResponseDecision,
    Decision,
    EndDecision,
    RouteDecision,
)
from llmling_agent.delegation.router import AgentRouter, CallbackRouter


class Ticket(BaseModel):
    """Support ticket with classification."""

    title: str
    description: str
    category: Literal["technical", "billing", "feature", "bug"]
    priority: Literal["low", "medium", "high", "critical"]
    needs_human_review: bool = False


class Resolution(BaseModel):
    """Support ticket resolution."""

    ticket_id: str
    solution: str
    resolved: bool
    followup_needed: bool = False
    assigned_to: str | None = None


AGENT_CONFIG = """
agents:
  classifier:
    type: pydantic_ai
    name: "Ticket Classifier"
    model: openai:gpt-4o-mini
    description: "Analyzes requests and creates structured tickets"
    system_prompts:
      - |
        You analyze support requests and create structured tickets.
        Consider urgency, category, and whether human review is needed.

  tech_support:
    type: pydantic_ai
    name: "Technical Support"
    model: openai:gpt-4o-mini
    system_prompts:
      - |
        You handle technical support tickets.
        Provide clear solutions and mark if followup is needed.

  billing:
    type: pydantic_ai
    name: "Billing Support"
    model: openai:gpt-4o-mini
    system_prompts:
      - You handle billing and payment issues.

  human_agent:
    type: human
    name: "Support Lead"
    description: "Human supervisor for complex cases"
    system_prompts:
      - You are an experienced support team lead.
"""


async def smart_router(ticket: Ticket, pool: AgentPool, router: AgentRouter) -> Decision:
    """Smart routing based on ticket properties."""
    # Critical tickets always go to human first
    if ticket.priority == "critical":
        reason = f"Critical {ticket.category} issue needs immediate attention"
        return AwaitResponseDecision(target_agent="human_agent", reason=reason)

    match (ticket.category, ticket.priority):
        case ("technical", "high"):
            # High priority tech issues get human review after tech support
            tech_agent = pool.get_agent("tech_support", return_type=Resolution)
            # Convert ticket to string for tech agent
            resolution = await tech_agent.run(str(ticket))
            if not resolution.data.resolved:
                reason = "Unresolved high-priority technical issue"
                return RouteDecision(target_agent="human_agent", reason=reason)
        case ("billing", "high" | "medium"):
            reason = "Priority billing issue"
            return AwaitResponseDecision(target_agent="billing", reason=reason)
        case _ if ticket.needs_human_review:
            reason = "Marked for human review"
            return RouteDecision(target_agent="human_agent", reason=reason)

    return EndDecision(reason="Ticket handled appropriately")


async def main(config_path: str):
    async with AgentPool[None](config_path) as pool:
        # Create type-safe agent
        classifier = pool.get_agent("classifier", return_type=Ticket)

        # Create smart controller
        controller = CallbackRouter[Ticket](pool, smart_router)
        # Process a support request
        request = (
            "I can't access my account and I have an urgent demo in 1 hour! "
            "This is blocking our whole team."
        )

        # Get structured ticket from classifier
        ticket_msg = await classifier.run(request)
        ticket = ticket_msg.data  # Type-safe Ticket

        print("\n[bold blue]Ticket Classification:[/]")
        print(f"Priority: {ticket.priority}")
        print(f"Category: {ticket.category}")
        print(f"Needs human review: {ticket.needs_human_review}")

        # Get routing decision
        decision = await controller.decide(ticket)

        match decision:
            case AwaitResponseDecision():
                print(f"\n[bold green]Routing to {decision.target_agent}[/]")
                print(f"Reason: {decision.reason}")

                # Get properly typed agent for resolution
                agent = pool.get_agent(decision.target_agent, return_type=Resolution)
                # Convert ticket to string
                response = await agent.run(str(ticket))
                resolution = response.data  # Type-safe Resolution

                print("\n[bold blue]Resolution:[/]")
                print(f"Solution: {resolution.solution}")
                print(f"Resolved: {resolution.resolved}")
                if resolution.followup_needed:
                    print(f"Followup needed with: {resolution.assigned_to}")

            case RouteDecision():
                print(f"\n[bold yellow]Forwarding to {decision.target_agent}[/]")
                print(f"Reason: {decision.reason}")
                next_agent = pool.get_agent(decision.target_agent)
                next_agent.outbox.emit(ticket_msg, None)

            case EndDecision():
                print("\n[bold green]Ticket handling complete[/]")
                print(f"Reason: {decision.reason}")


if __name__ == "__main__":
    import asyncio
    import tempfile

    from rich import print  # noqa: A004

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
