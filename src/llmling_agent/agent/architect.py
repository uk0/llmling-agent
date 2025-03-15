from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from upathtools import read_folder_as_text, read_path

from llmling_agent import Agent, models
from llmling_agent.common_types import YAMLCode
import llmling_agent_config


if TYPE_CHECKING:
    from llmling_agent.agent.agent import AgentType
    from llmling_agent.agent.structured import StructuredAgent


EXAMPLE = """
# Example agent with team
agents:
  analyzer:
    name: "Analyzer"
    model: "gpt-4"
    capabilities:
      can_load_resources: true

teams:
  analysis_team:
    mode: "sequential"
    members: ["analyzer"]
    connections:
      - target: "output_handler"
        type: "forward"
"""

SYS_PROMPT = """
You are an expert at creating LLMling-agent configurations.
Generate complete, valid YAML that can include:
- Agent configurations with appropriate tools and capabilities
- Team definitions with proper member relationships
- Connection setups for message routing
Follow the provided JSON schema exactly.
Only add stuff asked for by the user.
ONLY RETURN THE ACTUAL YAML. Your Output should ALWAYS be parseable by a YAML parser.
Nver answer with anything else. Dont prepend any sentences. Just return plain YAML.
"""


CONFIG_PATH = pathlib.Path(llmling_agent_config.__file__).parent
CORE_CONFIG_PATH = pathlib.Path(models.__file__).parent
README_URL = (
    "https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/README.md"
)


async def create_architect_agent(
    name: str = "config_generator",
    model: str = "copilot:claude-3.5-sonnet",
    provider: AgentType = "pydantic_ai",
) -> StructuredAgent[None, YAMLCode]:
    code = await read_folder_as_text(CONFIG_PATH, pattern="**/*.py")
    core_code = await read_folder_as_text(CORE_CONFIG_PATH, pattern="**/*.py")
    readme = await read_path(README_URL)
    context = f"Code:\n{core_code}\n{code}\n\nExample:\n{EXAMPLE}\n\\Readme:\n{readme}"
    agent = Agent[None](
        name,
        model=model,
        provider=provider,
        system_prompt=SYS_PROMPT,
    ).to_structured(YAMLCode)
    agent.conversation.add_context_message(context)
    return agent


if __name__ == "__main__":
    import asyncio

    agent = asyncio.run(create_architect_agent())
    result = agent.run_sync("write a config")
    print(result.content.code)
