"""Example of using agents as tools for downloads."""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from llmling_agent.delegation import AgentPool


if TYPE_CHECKING:
    from llmling_agent.agent import LLMlingAgent

MANIFEST_PATH = "src/llmling_agent_examples/download_agents.yml"
FILE_URL = "http://speedtest.tele2.net/10MB.zip"
PROMPT = f"Download this file using both agent tools available to you: {FILE_URL}"


async def main():
    async with AgentPool.open(MANIFEST_PATH) as pool:
        # Get the boss agent
        boss: LLMlingAgent[Any, Any] = pool.get_agent("overseer")
        # Register both downloaders as worker tools
        worker_1 = pool.get_agent("file_getter_1")
        boss.register_worker(worker_1)
        worker_2 = pool.get_agent("file_getter_2")
        boss.register_worker(worker_2)
        print("Sequential downloads:")
        start_time = time.time()
        result = await boss.run(PROMPT)
        duration = time.time() - start_time
        print(f"Sequential time: {duration:.2f} seconds")
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())
