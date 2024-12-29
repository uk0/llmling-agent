"""Example of using agents as tools for downloads."""

import asyncio
import tempfile
import time
from typing import TYPE_CHECKING, Any

import yamling

from llmling_agent.delegation import AgentPool


if TYPE_CHECKING:
    from llmling_agent.agent import LLMlingAgent

FILE_URL = "http://speedtest.tele2.net/10MB.zip"

# Agent configuration
AGENT_CONFIG = """\
agents:
  file_getter_1:
    name: "File Downloader 1"
    description: "Downloads files from URLs"
    model: openai:gpt-4o-mini
    system_prompts:
      - "You have ONE job: use the download_file tool to download files."
    environment:
      type: inline
      config:
        tools:
          download_file:
            import_path: llmling_agent_tools.download_file
            description: "Download file from URL to local path"

  overseer:
    name: "Download Coordinator"
    description: "Coordinates parallel downloads"
    model: openai:gpt-4o-mini
    role: overseer
    system_prompts:
      - |
        You coordinate file downloads using available agents. Your job is to:
        1. Use delegate_to to assign download tasks to file_getter_1 and file_getter_2
        2. Report the EXACT download results from the agents including speeds and sizes
"""

PROMPT = f"Download this file using both agent tools available to you: {FILE_URL}"


async def run_with_config(config_path: str):
    async with AgentPool.open(config_path) as pool:
        # Get the boss agent
        boss: LLMlingAgent[Any, Any] = pool.get_agent("overseer")

        # Create second downloader by cloning the first
        worker_1 = pool.get_agent("file_getter_1")
        worker_2 = await pool.clone_agent(worker_1, new_name="file_getter_2")

        # Register both as worker tools
        boss.register_worker(worker_1)
        boss.register_worker(worker_2)

        print("Calling both tools sequentally:")
        start_time = time.time()
        result = await boss.run(PROMPT)
        duration = time.time() - start_time
        print(f"Sequential time: {duration:.2f} seconds")
        print(result.data)


async def main():
    # Keep tempfile alive until we're done with the pool
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        yaml_config = yamling.load_yaml(AGENT_CONFIG)
        yamling.dump_yaml(yaml_config, stream=tmp)
        tmp.flush()
        await run_with_config(tmp.name)


if __name__ == "__main__":
    asyncio.run(main())
