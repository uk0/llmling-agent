"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from llmling.config.runtime import RuntimeConfig
from pydantic import BaseModel

from llmling_agent import LLMlingAgent
from llmling_agent.log import get_logger
from llmling_agent.models import AgentsManifest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.result import RunResult


logger = get_logger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for running agents."""

    agent_names: list[str]
    """Names of agents to run"""

    prompts: list[str]
    """List of prompts to send to the agent(s)"""

    mode: Literal["sequential", "parallel"] = "sequential"
    """Execution mode for multiple agents"""

    model: str | None = None
    """Optional model override"""


class AgentResponse(BaseModel):
    """Response from a single agent of a team."""

    agent_name: str
    """Name of the responding agent"""

    response: str
    """Agent's response"""

    success: bool
    """Whether the agent completed successfully"""

    error: str | None = None
    """Error message if agent failed"""


class AgentPool:
    """Pool of initialized agents.

    Each agent maintains its own runtime environment based on its configuration.
    """

    def __init__(
        self,
        manifest: AgentsManifest,
        *,
        agents_to_load: list[str] | None = None,
    ) -> None:
        """Initialize agent pool."""
        self.agents: dict[str, LLMlingAgent[Any, Any]] = {}
        self.manifest = manifest

        # Initialize requested agents
        available = set(manifest.agents)
        to_load = set(agents_to_load) if agents_to_load else available
        if invalid := (to_load - available):
            msg = f"Unknown agents: {', '.join(invalid)}"
            raise ValueError(msg)

        self._agents_to_load = to_load

    async def clone_agent[TDeps, TResult](
        self,
        agent: LLMlingAgent[TDeps, TResult] | str,
        new_name: str | None = None,
        *,
        model_override: str | None = None,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> LLMlingAgent[TDeps, TResult]:
        """Create a copy of an agent.

        Args:
            agent: Agent instance or name to clone
            new_name: Optional name for the clone
            model_override: Optional different model
            system_prompts: Optional different prompts
            template_context: Variables for template rendering

        Returns:
            The new agent instance
        """
        # Get original config
        if isinstance(agent, str):
            if agent not in self.manifest.agents:
                msg = f"Agent {agent} not found"
                raise KeyError(msg)
            config = self.manifest.agents[agent]
            original_agent: LLMlingAgent[TDeps, TResult] = await self.get_agent(agent)
        else:
            config = agent._context.config  # type: ignore
            original_agent = agent

        # Create new config
        new_config = config.model_copy(deep=True)

        # Apply overrides
        if model_override:
            new_config.model = model_override
        if system_prompts:
            new_config.system_prompts = system_prompts

        # Handle template rendering
        if template_context:
            new_config.system_prompts = new_config.render_system_prompts(template_context)

        # Create new agent with same runtime
        new_agent = LLMlingAgent[TDeps, TResult](
            runtime=original_agent._runtime,
            context=original_agent._context,
            result_type=original_agent._pydantic_agent._result_type,  # type: ignore
            model=new_config.model,  # type: ignore
            system_prompt=new_config.system_prompts,
            name=new_name or f"{config.name}_copy_{len(self.agents)}",
        )

        # Register in pool
        agent_name = new_agent.name
        self.manifest.agents[agent_name] = new_config
        self.agents[agent_name] = new_agent

        return new_agent

    async def get_agent[TDeps, TResult](
        self,
        name: str,
        *,
        deps_type: type[TDeps] | None = None,
        result_type: type[TResult] | None = None,
        model_override: str | None = None,
    ) -> LLMlingAgent[TDeps, TResult]:
        """Get or create a typed agent."""
        if name in self.agents:
            return self.agents[name]
        if name not in self._agents_to_load:
            msg = f"Agent {name} not in initialized set"
            raise KeyError(msg)

        config = self.manifest.agents[name]

        # Create runtime from agent's config
        cfg = config.get_config()
        async with RuntimeConfig.open(cfg) as runtime:
            new_agent: LLMlingAgent[TDeps, TResult] = LLMlingAgent(
                runtime=runtime,
                result_type=result_type or None,
                model=model_override or config.model,  # type: ignore
                system_prompt=config.system_prompts,
                name=name,
            )
            self.agents[name] = new_agent

        return self.agents[name]

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config_path: str,
        *,
        agents: list[str] | None = None,
    ) -> AsyncIterator[AgentPool]:
        """Open an agent pool from configuration.

        Args:
            config_path: Path to agent configuration file
            agents: Optional list of agent names to initialize.
                   If None, all agents from manifest are loaded.
        """
        manifest = AgentsManifest.from_file(config_path)
        pool = cls(manifest, agents_to_load=agents)
        try:
            yield pool
        finally:
            await pool.cleanup()

    async def run(
        self,
        config: ExecutionConfig,
    ) -> dict[str, list[RunResult[Any]]]:
        """Run agents according to configuration.

        Args:
            config: Execution configuration

        Returns:
            Dictionary mapping agent names to their results
        """
        match config.mode:
            case "sequential":
                return await self._run_sequential(config)
            case "parallel":
                return await self._run_parallel(config)
            case _:
                msg = f"Invalid execution mode: {config.mode}"
                raise ValueError(msg)

    async def _run_sequential(
        self,
        config: ExecutionConfig,
    ) -> dict[str, list[RunResult[Any]]]:
        """Run agents sequentially."""
        results = {}
        for name in config.agent_names:
            agent: LLMlingAgent[Any, Any] = await self.get_agent(
                name,
                model_override=config.model,
            )
            agent_results = []
            for prompt in config.prompts:
                result = await agent.run(prompt)
                agent_results.append(result)
            results[name] = agent_results
        return results

    async def _run_parallel(
        self,
        config: ExecutionConfig,
    ) -> dict[str, list[RunResult[Any]]]:
        """Run agents in parallel."""

        async def run_agent(name: str) -> list[RunResult[Any]]:
            agent: LLMlingAgent[Any, Any] = await self.get_agent(
                name,
                model_override=config.model,
            )
            return [await agent.run(p) for p in config.prompts]

        tasks = [run_agent(name) for name in config.agent_names]
        results = await asyncio.gather(*tasks)
        return dict(zip(config.agent_names, results))

    async def team_task(
        self,
        prompt: str,
        team: Sequence[str],
        *,
        mode: Literal["parallel", "sequential"] = "parallel",
    ) -> list[AgentResponse]:
        """Execute a task with a team of agents.

        Args:
            prompt: Task to execute
            team: List of agent names to collaborate
            mode: Whether to run agents in parallel or sequence

        Returns:
            List of responses from team members
        """

        async def run_agent(name: str) -> AgentResponse:
            try:
                agent: LLMlingAgent[Any, str] = await self.get_agent(name)
                result = await agent.run(prompt)
                response = str(result.data)
                return AgentResponse(agent_name=name, response=response, success=True)
            except Exception as e:
                logger.exception("Agent %s failed", name)
                return AgentResponse(
                    agent_name=name, response="", success=False, error=str(e)
                )

        if mode == "parallel":
            tasks = [run_agent(name) for name in team]
            return list(await asyncio.gather(*tasks))

        # Sequential execution
        return [await run_agent(name) for name in team]

    async def brainstorm(
        self,
        prompt: str,
        team: Sequence[str],
        *,
        rounds: int = 3,
    ) -> list[str]:
        """Collaborative brainstorming session.

        Args:
            prompt: Topic to brainstorm
            team: List of participating agents
            rounds: Number of brainstorming rounds

        Returns:
            List of ideas generated
        """
        ideas: list[str] = []

        for round_num in range(rounds):
            round_prompt = (
                f"Round {round_num + 1}/{rounds}. Previous ideas:\n"
                f"{'. '.join(ideas)}\n\n"
                f"Original task: {prompt}\n"
                "Please build on these ideas or add new ones."
            )

            responses = await self.team_task(round_prompt, team)
            ideas = [resp.response for resp in responses if resp.success]
        return ideas

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.manifest.agents)

    async def cleanup(self) -> None:
        """Clean up pool resources."""
        # Clean up each agent's runtime
        for agent in self.agents.values():
            if agent._runtime:
                await agent._runtime.shutdown()


async def main():
    async with AgentPool.open("agents.yml") as pool:
        overseer: LLMlingAgent[Any, str] = await pool.get_agent("overseer")
        from llmling_agent.delegation.tools import register_delegation_tools

        # Register all delegation tools
        register_delegation_tools(overseer, pool)

        # Now the overseer can use any delegation tool
        result = await overseer.run("""
            Please coordinate a team analysis:
            1. Use brainstorm with the development team
            2. Have the critic review the ideas
            3. Start a debate about the best approach
        """)
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
