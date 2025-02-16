"""State management for the web interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent.log import get_logger
from llmling_agent_web.type_utils import ChatHistory  # noqa: TC001


logger = get_logger(__name__)


@dataclass
class AgentState:
    """Manages state for the web interface."""

    agent_def: AgentsManifest
    """Loaded agent definition"""

    pool: AgentPool | None = None
    """Currently active agent runner"""

    history: dict[str, ChatHistory] = field(default_factory=dict)
    """Chat history for each agent"""

    @classmethod
    async def create(cls, file_path: str) -> AgentState:
        """Create a new agent state from a config file.

        Args:
            file_path: Path to agent configuration file

        Returns:
            Initialized agent state

        Raises:
            ValueError: If file cannot be loaded
        """
        try:
            logger.debug("Loading agent definition from: %s", file_path)
            agent_def = AgentsManifest.from_file(file_path)
            return cls(agent_def=agent_def)
        except Exception as e:
            error_msg = f"Failed to load agent file: {e}"
            raise ValueError(error_msg) from e

    def __str__(self) -> str:
        """String representation for logging."""
        return f"AgentState(agents={list(self.agent_def.agents.keys())})"

    async def select_agent(
        self,
        agent_name: str,
        model: str | None = None,
    ):
        """Select and initialize an agent.

        Args:
            agent_name: Name of agent to select
            model: Optional model override

        Raises:
            ValueError: If agent cannot be initialized
        """
        try:
            # Clean up existing pool
            await self.cleanup()

            # Create new pool
            self.pool = AgentPool(self.agent_def)
            if model:  # Apply model override if specified
                agent = self.pool.get_agent(agent_name)
                agent.set_model(model)  # type: ignore
            # Initialize history for this agent if needed
            if agent_name not in self.history:
                self.history[agent_name] = []
        except Exception as e:
            error_msg = f"Failed to initialize agent: {e}"
            raise ValueError(error_msg) from e

    async def cleanup(self):
        """Clean up resources."""
        if self.pool:
            await self.pool.cleanup()
            self.pool = None
