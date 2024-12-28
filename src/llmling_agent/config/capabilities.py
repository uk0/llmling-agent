"""Agent capabilities definition."""

from __future__ import annotations

from typing import Literal

from psygnal import EventedModel
from pydantic import ConfigDict


class Capabilities(EventedModel):
    """Defines what operations an agent is allowed to perform.

    Controls an agent's permissions and access levels including:
    - Agent discovery and delegation abilities
    - History access permissions
    - Statistics viewing rights
    - Tool usage restrictions

    Can be defined per role or customized per agent.
    """

    can_list_agents: bool = False
    """Whether the agent can discover other available agents."""

    can_delegate_tasks: bool = False
    """Whether the agent can delegate tasks to other agents."""

    can_observe_agents: bool = False
    """Whether the agent can monitor other agents' activities."""

    history_access: Literal["none", "own", "all"] = "none"
    """Level of access to conversation history.

    Levels:
    - none: No access to history
    - own: Can only access own conversations
    - all: Can access all agents' conversations
    """

    stats_access: Literal["none", "own", "all"] = "none"
    """Level of access to usage statistics.

    Levels:
    - none: No access to statistics
    - own: Can only view own statistics
    - all: Can view all agents' statistics
    """

    def enable(self, capability: str):
        """Enable a capability."""
        setattr(self, capability, True)

    def disable(self, capability: str):
        """Disable a capability."""
        setattr(self, capability, False)

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)
