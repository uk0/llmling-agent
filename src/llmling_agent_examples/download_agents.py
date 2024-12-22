from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, HttpUrl

from llmling_agent.delegation import AgentPool
from llmling_agent.delegation.tools import register_delegation_tools


if TYPE_CHECKING:
    from llmling_agent.agent.agent import LLMlingAgent

MANIFEST_PATH = "src/llmling_agent_examples/download_agents.yml"

FILE_URL = "http://speedtest.tele2.net/10MB.zip"  # Usually reliable
# or
# FILE_URL = "http://speedtest.ftp.otenet.gr/files/test100k.db"  # Another option
# or for a really small test
# FILE_URL = "http://example.com/"  # Just to test the download mechanism
# FILE_URL = "https://speed.hetzner.de/100MB.bin"


class DownloadRequest(BaseModel):
    url: HttpUrl
    target_path: Path


class DownloadResult(BaseModel):
    success: bool
    file_path: Path | None = None
    duration_ms: int | None = None
    error: str | None = None


async def main():
    async with AgentPool.open(MANIFEST_PATH) as pool:
        # Sequential download
        print("Sequential downloads:")
        start_time = time.time()

        responses = await pool.team_task(
            f"Download this file: {FILE_URL}",
            team=["file_getter_1", "file_getter_2"],
            mode="sequential",  # One after another
        )

        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")

        # Parallel download
        print("\nParallel downloads:")
        start_time = time.time()

        responses = await pool.team_task(
            f"Download this file: {FILE_URL}",
            team=["file_getter_1", "file_getter_2"],
            mode="parallel",  # Both at the same time
        )
        print(responses)
        parallel_time = time.time() - start_time
        print(f"Parallel time: {parallel_time:.2f} seconds")

        print(f"\nParallel was {sequential_time / parallel_time:.1f}x faster")

        # Or let the overseer handle it
        overseer: LLMlingAgent[Any, str] = await pool.get_agent("overseer")
        register_delegation_tools(overseer, pool)

        result = await overseer.run(f"""
            Please coordinate downloading this file twice:
            URL: {FILE_URL}

            1. Delegate to file_getter_1 and file_getter_2
            2. Have them work in parallel
            3. Report the results
        """)
        print("\nOverseer's report:")
        print(result.data)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
