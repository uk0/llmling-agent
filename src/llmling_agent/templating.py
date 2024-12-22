from __future__ import annotations

from typing import Any

from jinja2 import Environment


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
    env = Environment(autoescape=True, keep_trailing_newline=True)
    tpl = env.from_string(template)
    return tpl.render(agent=agent_context)
