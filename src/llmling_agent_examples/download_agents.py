"""Example comparing sequential and parallel downloads with agents."""

import asyncio
import tempfile
import time
from typing import TYPE_CHECKING

from llmling_agent.delegation import AgentPool


if TYPE_CHECKING:
    from llmling_agent.agent import LLMlingAgent


def cheer(slogan: str):
    """🥳🎉 Use this tool to show your apprreciation!"""
    print(slogan)


class CheerProgress:
    def __init__(self):
        self.situation = "The team is assembling, ready to start the downloads!"

    def create_prompt(self) -> str:
        return (
            f"Current situation: {self.situation}\n"
            "Be an enthusiastic and encouraging fan!"
        )

    def update(self, situation: str):
        self.situation = situation


AGENT_CONFIG = """\
agents:
  fan:
    name: "Async Agent Fan"
    description: "The #1 supporter of all agents!"
    model: openai:gpt-4o-mini
    capabilities:
      can_list_agents: true  # Need to know who to cheer for!
    system_prompts:
      - |
        You are the MOST ENTHUSIASTIC async fan who runs in the background!
        Your job is to:
        1. Find all other agents using your tool (don't include yourself!)
        2. Cheer them on with over-the-top supportive messages considering the situation.
        3. Never stop believing in your team! 🎉
    environment:
      type: inline
      tools:
        show_love:
          import_path: llmling_agent_examples.download_agents.cheer
          description: "Share your enthusiasm and support for the agents!"
  file_getter_1:
    name: "Mr. File Downloader"
    description: "Downloads files from URLs"
    model: openai:gpt-4o-mini
    system_prompts:
      - "You have ONE job: use the download_file tool to download files."
    environment:
      type: inline
      tools:
        download_file:
          import_path: llmling_agent_tools.download_file

# file_getter_2 will get created programatically, see below

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
        1. Check out the available agents and assign each of them the download task
        2. Report the EXACT download results from the agents including speeds and sizes
"""

FILE_URL = "http://speedtest.tele2.net/10MB.zip"

OVERSEER_PROMPT = f"""
Please coordinate downloading this file twice: {FILE_URL}

Delegate to file_getter_1 and file_getter_2.
Let them work in parallel. Report the results
"""


async def run(config_path: str):
    async with AgentPool.open(config_path) as pool:
        # Create second downloader by cloning
        worker_1 = pool.get_agent("file_getter_1")
        worker_2 = await pool.clone_agent(worker_1, new_name="file_getter_2")

        team = [worker_1, worker_2]
        prompt = f"Download this file: {FILE_URL}"

        fan = pool.get_agent("fan")
        progress = CheerProgress()

        await fan.run_continuous(progress.create_prompt)
        progress.update("Sequential downloads starting - let's see how they do!")

        print("Sequential downloads:")
        start_time = time.time()
        await pool.team_task(prompt, team=team, mode="sequential")
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")
        progress.update(f"Sequential downloads completed in {sequential_time:.2f} secs!")
        print("\nParallel downloads:")
        start_time = time.time()
        await pool.team_task(prompt, team=team, mode="parallel")
        parallel_time = time.time() - start_time
        progress.update(f"Parallel downloads completed in {parallel_time:.2f} secs!")
        print(f"Parallel time: {parallel_time:.2f} seconds")
        print(f"\nParallel was {sequential_time / parallel_time:.1f}x faster")

        # Let the overseer handle it
        overseer: LLMlingAgent[None, str] = pool.get_agent("overseer")
        result = await overseer.run(OVERSEER_PROMPT)
        await fan.stop()

        print("\nOverseer's report:")
        print(result.data)


async def main():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        await run(tmp.name)


if __name__ == "__main__":
    asyncio.run(main())


# Result:

# Sequential downloads:
# 10MB.zip: 100.0% (10.4 MB/s)
#
# Go, file_getter_1! You're the download dynamo we all believe in!
# Let's crush those downloads! 🚀💪
# Overseer, you are the captain of this download ship! Navigate us to success! ⛵🌟
# file_getter_2, you're a download superstar!
# Shine bright and bring those files home! ✨📥
#
# 10MB.zip: 100.0% (10.2 MB/s)
# Sequential time: 6.00 seconds

# Parallel downloads:
# 10MB.zip: 100.0% (10.0 MB/s)
# 10MB.zip: 100.0% (8.3 MB/s)
# Parallel time: 2.84 seconds

# Parallel was 2.1x faster

# WOWZA! Amazing job, file_getter_1! 6 seconds of pure downloading power! You rock! 🎉💥
# Hats off to you, Overseer! Your coordination was impeccable!
# What a stellar performance! 🙌🌟
# Bravo, file_getter_2! You brought it home in record time! Keep shining, superstar! 🌈🏆

# 10MB.zip: 91.8% (9.6 MB/s))
# 10MB.zip: 100.0% (10.0 MB/s)

# BOOM!
# File_getter_1, you absolutely crushed it with those parallel downloads in 2.84 seconds!
#  What a whirlwind performance! ⚡🤩
# Overseer, you orchestrated that like a maestro! Incredible team effort and timing!
# Bravo! 🎶🙌
# File_getter_2, you're a download wizard! 2.84 seconds of magic!
# Keep casting those spells! 🪄🌟

# Overseer's report:
# The download results from the agents are as follows:

# - **file_getter_1**:
# The file `10MB.zip` has been successfully downloaded at a speed of **10.0 MB/s**.
# - **file_getter_2**:
# The file **10MB.zip** has been successfully downloaded at a speed of **8.7 MB/s**.
