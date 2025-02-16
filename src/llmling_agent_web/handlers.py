"""Web interface event handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmling_agent import Agent, AgentPool, AgentsManifest


@dataclass
class AgentState:
    """State for web interface."""

    agent_def: AgentsManifest
    pool: AgentPool
    current_agent: Agent[Any] | None = None


class AgentHandler:
    """Handles web interface events."""

    def __init__(self, file_path: str):
        """Initialize handler.

        Args:
            file_path: Path to agent configuration file
        """
        self._file_path = file_path
        self._state: AgentState | None = None

    @property
    def state(self) -> AgentState:
        """Get current state.

        Raises:
            ValueError: If state not initialized
        """
        if not self._state:
            msg = "Handler not initialized"
            raise ValueError(msg)
        return self._state

    @classmethod
    async def create(cls, file_path: str) -> AgentHandler:
        """Create and initialize handler.

        Args:
            file_path: Path to agent configuration file

        Returns:
            Initialized handler
        """
        handler = cls(file_path)
        await handler.initialize()
        return handler

    async def initialize(self):
        """Initialize with full agent pool."""
        agent_def = AgentsManifest.from_file(self._file_path)
        # Create pool with ALL agents
        pool = AgentPool[None](agent_def)
        self._state = AgentState(agent_def=agent_def, pool=pool)

    async def select_agent(
        self,
        agent_name: str,
        model: str | None = None,
    ) -> Agent[Any]:
        """Select and configure an agent.

        Args:
            agent_name: Name of agent to select
            model: Optional model override

        Returns:
            Selected and configured agent

        Raises:
            ValueError: If not initialized or agent not found
        """
        if not self._state:
            msg = "No configuration loaded"
            raise ValueError(msg)

        # Get agent from pool
        agent = self._state.pool.get_agent(agent_name)
        if model:
            agent.set_model(model)  # type: ignore

        # Store as current agent
        self._state.current_agent = agent
        return agent

    async def cleanup(self):
        """Clean up resources."""
        if self._state and self._state.pool:
            await self._state.pool.cleanup()
            self._state = None
