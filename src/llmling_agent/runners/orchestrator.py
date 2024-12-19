"""Orchestration for multiple agent runners."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.result import RunResult, Usage
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.runners.exceptions import AgentNotFoundError, NoPromptsError
from llmling_agent.runners.single import SingleAgentRunner


if TYPE_CHECKING:
    from llmling_agent.models import AgentsManifest
    from llmling_agent.runners.models import AgentRunConfig


logger = get_logger(__name__)
T = TypeVar("T", default=str)


class AgentOrchestrator[T]:
    """Orchestrates multiple agent runners."""

    def __init__(
        self,
        agent_def: AgentsManifest,
        run_config: AgentRunConfig,
    ) -> None:
        """Initialize orchestrator.

        Args:
            agent_def: Complete agent definition including responses
            run_config: Configuration for running agents
        """
        self.agent_def = agent_def
        self.run_config = run_config
        self.runners: dict[str, SingleAgentRunner] = {}

    def validate(self) -> None:
        """Validate configuration before running.

        Raises:
            AgentNotFoundError: If requested agent not found
            NoPromptsError: If no prompts provided
        """
        # Check agents exist
        missing = [
            name
            for name in self.run_config.agent_names
            if name not in self.agent_def.agents
        ]
        if missing:
            msg = f"Agent(s) not found: {', '.join(missing)}"
            raise AgentNotFoundError(msg)

        # Check we have prompts
        if not self.run_config.prompts:
            msg = "No prompts provided and no default prompts in configuration"
            raise NoPromptsError(msg)

    async def initialize_agent(
        self,
        agent_name: str,
        *,
        model_override: str | None = None,
    ) -> SingleAgentRunner:
        """Initialize a specific agent.

        Args:
            agent_name: Name of the agent to initialize
            model_override: Optional model override

        Returns:
            Initialized agent runner

        Raises:
            KeyError: If agent name not found
        """
        # Clean up existing runner if any
        await self.cleanup_agent(agent_name)

        # Create and initialize new runner
        config = self.agent_def.agents[agent_name]
        runner: SingleAgentRunner[Any] = SingleAgentRunner(
            agent_config=config,
            response_defs=self.agent_def.responses,
            model_override=model_override or self.run_config.model,
        )
        await runner.__aenter__()
        self.runners[agent_name] = runner
        return runner

    async def run_single(self) -> list[RunResult[T]]:
        """Execute a single agent with conversation support.

        Returns:
            List of results from the conversation
        """
        agent_name = self.run_config.agent_names[0]
        runner = await self.initialize_agent(agent_name)
        return await runner.run_conversation(self.run_config.prompts)

    async def run_multiple(self) -> dict[str, list[RunResult[T]]]:
        """Execute multiple agents in parallel with conversation support.

        Returns:
            Dict mapping agent names to their conversation results
        """
        # Initialize all agents first
        await asyncio.gather(*[
            self.initialize_agent(name) for name in self.run_config.agent_names
        ])

        # Run conversations in parallel
        tasks = [
            self.runners[name].run_conversation(self.run_config.prompts)
            for name in self.run_config.agent_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results to agent names, converting exceptions to error results
        return {
            name: (
                result
                if not isinstance(result, Exception)
                else [
                    RunResult(
                        _all_messages=[
                            ModelRequest(
                                parts=[UserPromptPart(content=f"Error: {result}")]
                            )
                        ],
                        _new_message_index=0,
                        data=f"Error: {result}",  # type: ignore[arg-type]
                        _usage=Usage(),
                    )
                ]
            )
            for name, result in zip(self.run_config.agent_names, results)
        }

    async def run(self) -> Any:
        """Run the agent(s) based on configuration.

        Returns:
            Single agent results or dict of results for multiple agents
        """
        self.validate()
        if len(self.run_config.agent_names) == 1:
            return await self.run_single()
        return await self.run_multiple()

    async def cleanup_agent(self, agent_name: str) -> None:
        """Clean up a specific agent's resources.

        Args:
            agent_name: Name of the agent to clean up
        """
        if runner := self.runners.pop(agent_name, None):
            await runner.__aexit__(None, None, None)

    async def cleanup(self) -> None:
        """Clean up all runners."""
        for runner in list(self.runners.values()):
            await runner.__aexit__(None, None, None)
        self.runners.clear()
