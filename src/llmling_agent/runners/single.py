"""Single agent runner implementation."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, TypeVar

from llmling.config.runtime import RuntimeConfig

from llmling_agent.agent import LLMlingAgent
from llmling_agent.log import get_logger
from llmling_agent.runners.exceptions import NotInitializedError


if TYPE_CHECKING:
    from types import TracebackType

    from pydantic_ai.result import RunResult

    from llmling_agent.models import AgentConfig
    from llmling_agent.responses import ResponseDefinition


logger = get_logger(__name__)
T = TypeVar("T")

RUNTIME_NOT_INITIALIZED = """
Runtime not initialized. Use 'async with' context manager:
async with SingleAgentRunner(...) as runner:
    result = await runner.run('prompt')
"""
AGENT_NOT_INITIALIZED = """
Agent not initialized. Use 'async with' context manager:
async with SingleAgentRunner(...) as runner:
    result = await runner.run('prompt')
"""


class SingleAgentRunner[T](AbstractAsyncContextManager):
    """Handles execution of a single agent with its own runtime environment."""

    def __init__(
        self,
        agent_config: AgentConfig,
        response_defs: dict[str, ResponseDefinition],
        *,
        model_override: str | None = None,
    ) -> None:
        """Initialize agent runner.

        Args:
            agent_config: Configuration for this agent
            response_defs: Available response type definitions
            model_override: Optional model override
        """
        self.agent_config = agent_config
        self.response_defs = response_defs
        self.model_override = model_override
        self._agent: LLMlingAgent[Any] | None = None
        self._runtime: RuntimeConfig | None = None

    @property
    def agent(self) -> LLMlingAgent[Any]:
        """Get the initialized agent.

        Returns:
            The initialized LLMlingAgent

        Raises:
            NotInitializedError: If accessed outside context manager
        """
        if self._agent is None:
            raise NotInitializedError(AGENT_NOT_INITIALIZED)
        return self._agent

    @property
    def runtime(self) -> RuntimeConfig:
        """Get the initialized runtime.

        Returns:
            The initialized RuntimeConfig

        Raises:
            NotInitializedError: If accessed outside context manager
        """
        if self._runtime is None:
            raise NotInitializedError(RUNTIME_NOT_INITIALIZED)
        return self._runtime

    async def __aenter__(self) -> SingleAgentRunner[T]:
        """Set up runtime and agent."""
        # Create runtime with agent's environment
        config = self.agent_config.get_config()
        runtime_cm = RuntimeConfig.open(config)
        self._runtime = await runtime_cm.__aenter__()

        # Create agent with potential model override
        if self.model_override:
            # might be worth checking out making the model unfrozen
            self.agent_config = self.agent_config.model_copy(
                update={"model": self.model_override}
            )

        # Initialize agent
        kwargs = self.agent_config.get_agent_kwargs()
        self._agent = LLMlingAgent(runtime=self.runtime, **kwargs)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up resources."""
        if self._runtime:
            await self._runtime.__aexit__(exc_type, exc_val, exc_tb)
            self._runtime = None
            self._agent = None

    async def run(self, prompt: str) -> RunResult[T]:
        """Run the agent with a prompt.

        Args:
            prompt: The prompt to send to the agent

        Returns:
            Agent's response

        Raises:
            NotInitializedError: If runner not initialized via context manager
        """
        return await self.agent.run(prompt)

    async def run_conversation(
        self,
        prompts: list[str],
    ) -> list[RunResult[T]]:
        """Run a conversation with multiple prompts.

        Args:
            prompts: List of prompts to send in sequence

        Returns:
            List of responses

        Raises:
            NotInitializedError: If runner not initialized via context manager
        """
        if not prompts:
            return []

        results: list[RunResult[T]] = []
        result = await self.run(prompts[0])
        results.append(result)

        for prompt in prompts[1:]:
            history = result.new_messages()
            result = await self.agent.run(prompt, message_history=history)
            results.append(result)

        return results
