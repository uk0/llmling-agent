"""Agent roles with predefined capabilities and prompts."""

from __future__ import annotations

from typing import Literal

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from pydantic import BaseModel, ConfigDict
import yamling

from llmling_agent.config.capabilities import Capabilities


class Role(BaseModel):
    """Defines a role and its capabilities.

    A role defines what an agent can do and how it should behave, including:
    - Access permissions and capabilities
    - System prompts that guide behavior
    - Description for documentation
    """

    name: str
    """Unique identifier for the role."""

    capabilities: Capabilities
    """Permissions and access levels for the role."""

    prompts: list[str]
    """System prompts that define the role's behavior and instructions."""

    description: str
    """Human-readable description of the role's purpose."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


# Built-in role definitions
BASIC = Role(
    name="basic",
    capabilities=Capabilities(),
    prompts=[],
    description="Basic agent with minimal capabilities",
)

OVERSEER = Role(
    name="overseer",
    capabilities=Capabilities(
        can_list_agents=True,
        can_delegate_tasks=True,
        can_observe_agents=True,
        history_access="all",
        stats_access="all",
    ),
    prompts=[
        """You are an overseer agent that coordinates with specialists.
        When you encounter tasks that could benefit from specific expertise:
        1. Use list_available_agents to discover specialists
        2. Use delegate_to when another agent would be more suitable
        3. Stop after delegating - the specialist will handle the request
        """,
    ],
    description="Coordinates task execution across multiple specialist agents",
)

SPECIALIST = Role(
    name="specialist",
    capabilities=Capabilities(
        history_access="own",
        stats_access="own",
    ),
    prompts=[
        """You are a specialist agent. While you can delegate tasks,
        prefer handling requests within your domain of expertise.
        Use list_available_agents and delegate_to only when you encounter
        tasks clearly outside your specialty.
        """,
    ],
    description="Expert in a specific domain with focused capabilities",
)

ASSISTANT = Role(
    name="assistant",
    capabilities=Capabilities(
        history_access="own",
        stats_access="none",
    ),
    prompts=[
        """You are a general assistant agent.
        You can delegate tasks to specialists when their expertise
        would provide better results.
        """,
    ],
    description="General purpose assistant with basic capabilities",
)


class RoleRegistry(BaseRegistry[str, Role]):
    """Registry for role definitions."""

    def __init__(self):
        """Initialize registry with built-in roles."""
        super().__init__()
        self._load_builtin_roles()

    def _load_builtin_roles(self) -> None:
        """Load built-in roles."""
        for role in [BASIC, OVERSEER, SPECIALIST, ASSISTANT]:
            self.register(role.name, role)

    @property
    def _error_class(self) -> type[exceptions.LLMLingError]:
        return exceptions.LLMLingError

    def _validate_item(self, item: Role) -> Role:
        return item

    def get_capabilities(self, role_name: str) -> Capabilities:
        """Get capabilities for a role."""
        return self[role_name].capabilities

    def get_prompts(self, role_name: str) -> list[str]:
        """Get system prompts for a role."""
        return self[role_name].prompts

    @classmethod
    def from_yaml(cls, path: str) -> RoleRegistry:
        """Create registry from YAML file."""
        registry = cls()
        data = yamling.load_yaml_file(path)

        for name, role_data in data.get("roles", {}).items():
            if "name" not in role_data:
                role_data["name"] = name
            role = Role.model_validate(role_data)
            registry.register(role.name, role)

        return registry


# Global registry instance
registry = RoleRegistry()


# Types
BuiltinRole = Literal["basic", "overseer", "specialist", "assistant"]
"""Built-in role types with predefined capabilities."""

RoleName = BuiltinRole | str
"""Valid role names, either built-in or custom."""


# Public API functions
def get_role_capabilities(role_name: str) -> Capabilities:
    """Get capabilities for a role name."""
    return registry.get_capabilities(role_name)


def get_role_prompts(role_name: str) -> list[str]:
    """Get system prompts for a role."""
    return registry.get_prompts(role_name)


def get_available_roles() -> list[str]:
    """Get names of all available roles."""
    return list(registry.keys())
