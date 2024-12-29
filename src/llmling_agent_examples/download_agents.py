"""Example comparing sequential and parallel downloads with agents."""

import asyncio
import tempfile
import time
from typing import TYPE_CHECKING, Any

from llmling_agent.delegation import AgentPool


if TYPE_CHECKING:
    from llmling_agent.agent import LLMlingAgent

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
    capabilities:
      can_delegate_tasks: true
      can_list_agents: true
    system_prompts:
      - |
        You coordinate file downloads using available agents. Your job is to:
        1. Use delegate_to to assign download tasks to file_getter_1 and file_getter_2
        2. Report the EXACT download results from the agents including speeds and sizes
"""

FILE_URL = "http://speedtest.tele2.net/10MB.zip"

OVERSEER_PROMPT = f"""
Please coordinate downloading this file twice:
URL: {FILE_URL}

1. Delegate to file_getter_1 and file_getter_2
2. Have them work in parallel
3. Report the results
"""


async def run(config_path: str):
    async with AgentPool.open(config_path) as pool:
        # Create second downloader by cloning
        worker_1 = pool.get_agent("file_getter_1")
        worker_2 = await pool.clone_agent(worker_1, new_name="file_getter_2")

        team = [worker_1, worker_2]
        prompt = f"Download this file: {FILE_URL}"

        print("Sequential downloads:")
        start_time = time.time()
        responses = await pool.team_task(prompt, team=team, mode="sequential")
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")

        print("\nParallel downloads:")
        start_time = time.time()
        responses = await pool.team_task(prompt, team=team, mode="parallel")
        print(responses)
        parallel_time = time.time() - start_time
        print(f"Parallel time: {parallel_time:.2f} seconds")
        print(f"\nParallel was {sequential_time / parallel_time:.1f}x faster")

        # Let the overseer handle it
        overseer: LLMlingAgent[Any, Any] = pool.get_agent("overseer")
        result = await overseer.run(OVERSEER_PROMPT)
        print("\nOverseer's report:")
        print(result.data)


async def main():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        await run(tmp.name)


if __name__ == "__main__":
    asyncio.run(main())
