"""State management for the web interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from llmling import Config, RuntimeConfig
from upath import UPath

from llmling_agent.agent import LLMlingAgent
from llmling_agent.models import AgentDefinition


logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Manages state for the web interface."""

    agent_def: AgentDefinition
    """Loaded agent definition"""

    runtime: RuntimeConfig
    """Runtime configuration"""

    current_agent: LLMlingAgent[str] | None = None
    """Currently active agent instance"""

    history: dict[str, list[list[str]]] = field(default_factory=dict)
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
            path = str(UPath(file_path))
            logger.debug("Loading agent definition from: %s", path)
            agent_def = AgentDefinition.from_file(path)
            logger.debug("Loaded agent definition: %s", agent_def)
            # Create Config object first if needed
            logger.debug("Creating runtime config")
            config = Config()
            logger.debug("Initializing runtime")
            runtime = RuntimeConfig.from_config(config)
            await runtime.__aenter__()
            return cls(agent_def=agent_def, runtime=runtime)
        except Exception as e:
            error_msg = f"Failed to load agent file: {e}"
            raise ValueError(error_msg) from e

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
            # Get agent config and apply model override
            config = self.agent_def.agents[agent_name]
            if model:
                config.model = model

            # Initialize agent
            self.current_agent = LLMlingAgent(
                runtime=self.runtime, name=agent_name, **config.model_dump()
            )

            # Initialize history for this agent if needed
            if agent_name not in self.history:
                self.history[agent_name] = []

        except Exception as e:
            error_msg = f"Failed to initialize agent: {e}"
            raise ValueError(error_msg) from e

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.runtime.__aexit__(None, None, None)
