"""Run command for agent execution."""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING, Any

from llmling.cli.constants import verbose_opt
from pydantic import ValidationError
import typer as t


if TYPE_CHECKING:
    from llmling_agent.agent.agent import LLMlingAgent

PROMPT_HELP = "Include named prompt from environment configuration"
OUTPUT_HELP = "Output format (text/json/yaml)"


def run_command(
    agent_name: str = t.Argument(help="Agent name(s) to run (can be comma-separated)"),
    prompts: list[str] = t.Argument(None, help="Additional prompts to send"),  # noqa: B008
    config_path: str = t.Option(
        None, "-c", "--config", help="Override agent configuration path"
    ),
    include_prompt: list[str] = t.Option(  # noqa: B008
        None,
        "--include-prompt",
        "-p",
        help=PROMPT_HELP,
    ),
    environment: str = t.Option(
        None, "--environment", "-e", help="Override agent's environment"
    ),
    model: str = t.Option(None, "--model", "-m", help="Override agent's model"),
    output_format: str = t.Option("text", "-o", "--output-format", help=OUTPUT_HELP),
    verbose: bool = verbose_opt,
):
    """Run agent with prompts.

    First runs any prompts provided via --include-prompt,
    then any additional prompts provided as arguments.

    Examples:
        # Run with single prompt
        llmling-agent run myagent "Analyze this text"

        # Multiple agents
        llmling-agent run "agent1,agent2" "Process data"

        # Include environment prompt
        llmling-agent run myagent -p analyze_code

        # Multiple environment prompts
        llmling-agent run myagent -p greet -p analyze

        # Environment prompt + additional prompt
        llmling-agent run myagent -p analyze "And summarize it"

        # Override model
        llmling-agent run myagent -m gpt-4 "Complex analysis"
    """
    from llmling.cli.utils import format_output
    from llmling.config.runtime import RuntimeConfig

    from llmling_agent.delegation import AgentPool
    from llmling_agent.models import AgentsManifest
    from llmling_agent_cli import resolve_agent_config

    try:
        # Resolve configuration path
        try:
            config_path = resolve_agent_config(config_path)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        # Load and validate agent definition
        try:
            agent_def = AgentsManifest.from_file(config_path)
        except ValidationError as e:
            t.echo("Agent configuration validation failed:", err=True)
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                t.echo(f"  {location}: {error['msg']}", err=True)
            raise t.Exit(1) from e

        # Parse agent names
        agent_names = [name.strip() for name in agent_name.split(",")]

        # Check agents exist
        missing = [name for name in agent_names if name not in agent_def.agents]
        if missing:
            msg = f"Agent(s) not found: {', '.join(missing)}"
            raise t.BadParameter(msg)  # noqa: TRY301

        final_prompts: list[str] = []

        # 1. First, add ALL prompts from agent configs (always included)
        for name in agent_names:
            config = agent_def.agents[name]
            if config.user_prompts:
                final_prompts.extend(config.user_prompts)

        # 2. Add RuntimeConfig prompts if specified
        if include_prompt:

            async def get_env_prompts():
                async with RuntimeConfig.open(config_path) as runtime:
                    for prompt_name in include_prompt:
                        try:
                            messages = await runtime.render_prompt(prompt_name)
                            final_prompts.extend(
                                msg.get_text_content() for msg in messages
                            )
                        except Exception as e:
                            msg = f"Failed to load prompt {prompt_name}: {e}"
                            raise t.BadParameter(msg) from e

            asyncio.run(get_env_prompts())

        # 3. Add additional prompts provided as arguments
        if prompts:
            final_prompts.extend(prompts)

        if not final_prompts:
            msg = "No prompts available (neither in config nor provided)"
            raise t.BadParameter(msg)  # noqa: TRY301

        async def run():
            async with AgentPool.open(config_path, agents=agent_names) as pool:
                if environment:
                    # Set environment override for all agents
                    pool.manifest.agents = {
                        name: config.model_copy(update={"environment": environment})
                        for name, config in pool.manifest.agents.items()
                    }

                if len(agent_names) == 1:
                    # Single agent execution
                    agent: LLMlingAgent[Any, Any] = pool.get_agent(
                        agent_names[0], model_override=model
                    )
                    results = []
                    for prompt in final_prompts:
                        result = await agent.run(prompt)
                        if isinstance(result.data, str):
                            print(result.data)
                        else:
                            format_output(result.data, output_format)
                        results.append(result)
                else:
                    # Team task execution
                    for prompt in final_prompts:
                        responses = await pool.team_task(
                            prompt,
                            agent_names,
                            mode="parallel",
                            model_override=model,
                            environment_override=environment,
                        )
                        formatted = {r.agent_name: r.response for r in responses}
                        format_output(formatted, output_format)

        asyncio.run(run())

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
