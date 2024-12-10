"""Event handlers for the web interface."""

from __future__ import annotations

from llmling import Config
from llmling.config.runtime import RuntimeConfig
from upath import UPath

from llmling_agent.agent import LLMlingAgent
from llmling_agent.log import get_logger
from llmling_agent.models import AgentDefinition
from llmling_agent.web.state import AgentState


logger = get_logger(__name__)


class AgentHandler:
    """Handles web interface events."""

    def __init__(self, file_path: str) -> None:
        """Initialize handler.

        Args:
            file_path: Initial configuration file path
        """
        self._state: AgentState | None = None
        self._file_path = str(UPath(file_path))

    async def initialize(self) -> None:
        """Initialize the handler state."""
        self._state = await AgentState.create(self._file_path)

    @property
    def state(self) -> AgentState:
        """Get the current state.

        Returns:
            Current agent state

        Raises:
            RuntimeError: If state not initialized
        """
        if not self._state:
            msg = "Handler not initialized"
            raise RuntimeError(msg)
        return self._state

    async def load_agent_file(
        self,
        file_path: str,
    ) -> tuple[list[str], str]:
        """Handle agent file selection.

        Args:
            file_path: Path to agent configuration file

        Returns:
            Tuple of (agent choices, status message)
        """
        try:
            # Clean up old state if it exists
            path = str(UPath(file_path))
            logger.debug("Loading file from path: %s", path)

            if self._state:
                await self._state.cleanup()

            # Create new state
            logger.debug("Creating new state")
            self._state = await AgentState.create(path)
            logger.debug("Agent definition: %s", self._state.agent_def)
            agents = list(self._state.agent_def.agents)
            logger.debug("Found agents: %s", agents)
            msg = f"Loaded {len(agents)} agents from {path}"
            logger.info(msg)
        except Exception as e:
            error = f"Error loading file: {e}"
            logger.exception(error)
            return [], error
        else:
            return agents, msg

    async def select_agent(
        self,
        file_path: str | None,
        agent_name: str | None,
        model: str | None,
    ) -> tuple[str, list[list[str]]]:
        """Handle agent selection.

        Args:
            file_path: Current configuration file
            agent_name: Name of agent to select
            model: Optional model override

        Returns:
            Tuple of (status message, chat history)
        """
        if not file_path:
            msg = "No file path provided"
            raise ValueError(msg)

        if not agent_name:
            msg = "No agent name provided"
            raise ValueError(msg)

        try:
            # Initialize state if not already done
            agent_def = AgentDefinition.from_file(str(file_path))

            # Create empty config for now
            config = Config()
            runtime = RuntimeConfig.from_config(config)
            await runtime.__aenter__()
            self._state = AgentState(agent_def=agent_def, runtime=runtime)

            # Initialize agent with validation
            if agent_name not in agent_def.agents:
                msg = f"Agent '{agent_name}' not found in configuration"
                raise ValueError(msg)  # noqa: TRY301

            agent_config = agent_def.agents[agent_name]
            if model:
                agent_config.model = model
            kwargs = agent_config.get_agent_kwargs(name=agent_name)
            self._state.current_agent = LLMlingAgent(runtime=runtime, **kwargs)
        except Exception:
            if self._state and self._state.runtime:
                await self._state.runtime.__aexit__(None, None, None)
            self._state = None
            raise
        else:
            return f"Agent {agent_name} ready", []

    async def send_message(
        self,
        message: str,
        chat_history: list[list[str]],
    ) -> tuple[str, list[list[str]], str]:
        """Handle sending a chat message.

        Args:
            message: User message
            chat_history: Current chat history

        Returns:
            Tuple of (cleared message, updated history, status)
        """
        if not message.strip():
            return "", chat_history, "Message is empty"

        if not self.state.current_agent:
            return message, chat_history, "No agent selected"

        try:
            # Add user message to history
            new_history = list(chat_history)
            new_history.append([message, ""])  # Use list, not tuple

            # Get agent response
            result = await self.state.current_agent.run(message)
            response = result.data

            # Update history
            new_history[-1][1] = response  # Update the second element

            # Update stored history
            if self.state.current_agent and self.state.current_agent._name:
                self.state.history[self.state.current_agent._name] = new_history
        except Exception as e:
            error = f"Error getting response: {e}"
            logger.exception(error)
            new_history.pop()  # Remove failed message
            return message, new_history, error
        else:
            return "", new_history, "Message sent"
