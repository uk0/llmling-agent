"""State management for the web interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmling_agent.log import get_logger
from llmling_agent.models import AgentsManifest
from llmling_agent.runners import SingleAgentRunner
from llmling_agent.web.type_utils import ChatHistory  # noqa: TC001


logger = get_logger(__name__)


@dataclass
class AgentState:
    """Manages state for the web interface."""

    agent_def: AgentsManifest
    """Loaded agent definition"""

    current_runner: SingleAgentRunner[str] | None = None
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
            logger.debug("Loaded agent definition: %s", agent_def)
            return cls(agent_def=agent_def)
        except Exception as e:
            error_msg = f"Failed to load agent file: {e}"
            raise ValueError(error_msg) from e

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"AgentState(agents={list(self.agent_def.agents.keys())}, "
            f"has_runner={self.current_runner is not None})"
        )

    async def select_agent(
        self,
        agent_name: str,
        model: str | None = None,
    ) -> None:
        """Select and initialize an agent.

        Args:
            agent_name: Name of agent to select
            model: Optional model override

        Raises:
            ValueError: If agent cannot be initialized
        """
        try:
            # Clean up existing runner
            await self.cleanup()

            # Create new runner
            config = self.agent_def.agents[agent_name]
            runner = SingleAgentRunner[str](
                agent_config=config,
                response_defs=self.agent_def.responses,
                model_override=model,
            )
            # Initialize the runner using context manager
            await runner.__aenter__()
            self.current_runner = runner

            # Initialize history for this agent if needed
            if agent_name not in self.history:
                self.history[agent_name] = []

        except Exception as e:
            error_msg = f"Failed to initialize agent: {e}"
            raise ValueError(error_msg) from e

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.current_runner:
            await self.current_runner.__aexit__(None, None, None)
            self.current_runner = None
