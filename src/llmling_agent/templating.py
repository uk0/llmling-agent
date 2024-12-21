from __future__ import annotations

from typing import Any

from jinja2 import Environment


def create_template_env() -> Environment:
    """Create Jinja environment with safety settings."""
    return Environment(
        autoescape=True,  # Safety first
        keep_trailing_newline=True,
    )


def render_prompt(
    template: str,
    agent_context: dict[str, Any],
) -> str:
    """Render a prompt template with context.

    Available variables:
        agent.name: Name of the agent
        agent.id: Number of the clone (for cloned agents)
        agent.role: Role of the agent
        agent.model: Model name
    """
    env = create_template_env()
    tpl = env.from_string(template)
    return tpl.render(agent=agent_context)
