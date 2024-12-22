"""Event handlers for the web interface."""

from __future__ import annotations

from llmling_agent.log import get_logger
from llmling_agent_web.state import AgentState


logger = get_logger(__name__)


class AgentHandler:
    """Handles web interface events."""

    def __init__(self, file_path: str):
        """Initialize handler.

        Args:
            file_path: Initial configuration file path
        """
        self._file_path = file_path
        self._state: AgentState | None = None

    @classmethod
    async def create(cls, file_path: str) -> AgentHandler:
        """Create and initialize a new handler.

        Args:
            file_path: Path to configuration file

        Returns:
            Initialized handler

        Raises:
            ValueError: If initialization fails
        """
        handler = cls(file_path)
        await handler.initialize()
        return handler

    async def initialize(self):
        """Initialize the handler state.

        Raises:
            ValueError: If initialization fails
        """
        logger.debug("Initializing handler with file: %s", self._file_path)
        self._state = await AgentState.create(self._file_path)
        logger.debug("Handler initialized with state: %s", self._state)

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
            logger.debug("Loading file from path: %s", file_path)

            # Clean up old state if it exists
            if self._state:
                await self._state.cleanup()

            # Create new state
            self._state = await AgentState.create(file_path)
            agents = list(self._state.agent_def.agents)
            msg = f"Loaded {len(agents)} agents from {file_path}"
            logger.info(msg)

        except Exception as e:
            error = f"Error loading file: {e}"
            logger.exception(error)
            return [], error
        else:
            return agents, msg

    async def select_agent(
        self,
        agent_name: str | None,
        model: str | None,
    ) -> tuple[str, list[list[str]]]:
        """Handle agent selection."""
        if not agent_name:
            msg = "No agent name provided"
            raise ValueError(msg)

        try:
            await self.state.select_agent(agent_name, model)
        except Exception:
            if self._state:
                await self._state.cleanup()
            self._state = None
            raise
        else:
            return f"Agent {agent_name} ready", []

    async def send_message(
        self,
        message: str,
        chat_history: list[dict[str, str]],  # Note: Using dict format consistently
    ) -> tuple[str, list[dict[str, str]], str]:
        """Handle sending a chat message."""
        if not message.strip():
            return "", chat_history, "Message is empty"

        if not self.state.pool:
            return message, chat_history, "No agent selected"

        try:
            # Get agent from pool and send message
            agent = next(iter(self.state.pool.agents.values()))
            result = await agent.run(message)
            response = str(result.data)

            # Update history with new messages
            new_history = list(chat_history)
            new_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ])

            # Store updated history
            agent_name = agent.name
            self.state.history[agent_name] = [
                [msg["content"] for msg in pair]
                for pair in zip(new_history[::2], new_history[1::2])
            ]
        except Exception as e:
            error = f"Error getting response: {e}"
            logger.exception(error)
            return message, chat_history, error
        else:
            return "", new_history, "Message sent"
