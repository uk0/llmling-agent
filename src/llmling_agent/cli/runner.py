from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from llmling import Config
from llmling.cli.utils import format_output
from llmling.config.runtime import RuntimeConfig
import typer as t

from llmling_agent.factory import create_agents_from_config


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from pydantic_ai.result import RunResult

    from llmling_agent.agent import LLMlingAgent
    from llmling_agent.models import AgentDefinition


T = TypeVar("T")


@dataclass
class AgentRunConfig:
    """Configuration for running agents."""

    agent_names: list[str]
    """Names of agents to run"""

    prompts: list[str]
    """List of prompts to send to the agent(s)"""

    environment: str | None
    """Optional environment override path"""

    model: str | None
    """Optional model override"""

    output_format: str
    """Output format for results (text/json/yaml)"""


class AgentRunner:
    """Handles execution of one or multiple agents."""

    def __init__(
        self,
        agent_def: AgentDefinition,
        run_config: AgentRunConfig,
    ) -> None:
        """Initialize runner with configuration."""
        self.agent_def = agent_def
        self.config = run_config
        self.agents_to_run = {
            name: agent_def.agents[name] for name in run_config.agent_names
        }

    def validate(self) -> None:
        """Validate configuration before running.

        Raises:
            t.BadParameter: If configuration is invalid
        """
        # Check agents exist
        missing = [
            name for name in self.config.agent_names if name not in self.agent_def.agents
        ]
        if missing:
            msg = f"Agent(s) not found: {', '.join(missing)}"
            raise t.BadParameter(msg)

        # Check we have prompts
        if not self.config.prompts:
            msg = "No prompts provided and no default prompts in configuration"
            raise t.BadParameter(msg)

    async def run_single_agent(
        self,
        runtime: RuntimeConfig,
        agents: dict[str, LLMlingAgent[Any]],
    ) -> None:
        """Execute a single agent with conversation support."""
        agent = agents[self.config.agent_names[0]]
        result = await agent.run(self.config.prompts[0])
        format_output(result.data, self.config.output_format)

        for prompt in self.config.prompts[1:]:
            result = await agent.run(
                prompt,
                message_history=result.new_messages(),
            )
            format_output(result.data, self.config.output_format)

    async def run_multiple_agents(
        self,
        runtime: RuntimeConfig,
        agents: dict[str, LLMlingAgent[Any]],
    ) -> None:
        """Execute multiple agents in parallel with conversation support."""
        # Initial parallel execution
        tasks: list[Coroutine[Any, Any, RunResult[Any]]] = [
            agents[name].run(self.config.prompts[0]) for name in self.agents_to_run
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Type assertion to help type checker
        results_typed: list[RunResult[Any] | Exception] = results  # type: ignore

        self._format_parallel_results(dict(zip(self.agents_to_run, results_typed)))

        # Handle conversation
        for prompt in self.config.prompts[1:]:
            tasks = []
            active_agents = []
            for name, prev_result in zip(self.agents_to_run, results_typed):
                if isinstance(prev_result, Exception):
                    continue
                active_agents.append(name)
                agent = agents[name]
                msgs = prev_result.new_messages()
                result = agent.run(prompt, message_history=msgs)
                tasks.append(result)

            if not tasks:
                t.echo("All agents failed, stopping conversation")
                break

            results = await asyncio.gather(*tasks, return_exceptions=True)
            results_typed = results  # type: ignore
            self._format_parallel_results(dict(zip(active_agents, results_typed)))

    def _format_parallel_results(
        self,
        results: dict[str, RunResult[Any] | Exception],
    ) -> None:
        """Format results from parallel execution."""
        output = {
            name: (result.data if not isinstance(result, Exception) else str(result))
            for name, result in results.items()
        }
        format_output(output, self.config.output_format)

    async def run(self) -> None:
        """Execute the agent(s)."""
        # Apply model override if specified
        if self.config.model:
            for config in self.agents_to_run.values():
                config.model = self.config.model

        async with RuntimeConfig.open(self.config.environment or Config()) as runtime:
            agents = create_agents_from_config(self.agent_def, runtime)

            if len(self.agents_to_run) == 1:
                await self.run_single_agent(runtime, agents)
            else:
                await self.run_multiple_agents(runtime, agents)
