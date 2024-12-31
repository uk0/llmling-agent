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
        2. Cheer them on with over-the-top supportive messages
        3. Never stop believing in your team! 🎉
    environment:
      type: inline
      config:
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
      config:
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
        await fan.run_continuous("Start cheering!")

        print("Sequential downloads:")
        start_time = time.time()
        await pool.team_task(prompt, team=team, mode="sequential")
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")

        print("\nParallel downloads:")
        start_time = time.time()
        await pool.team_task(prompt, team=team, mode="parallel")
        parallel_time = time.time() - start_time
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
# 10MB.zip: 100.0% (4.9 MB/s)
# Go Team Fan! You're the enthusiasm we all need! 🎉🥳
# File Getter 1, your skills in retrieving files are unmatched! Keep rocking it! 🌟📂
# Overseer, you're the watchful eye keeping things in check! We believe in you! 💪👁️
# File Getter 2, keep those files coming! You're the best at what you do! 🚀📁
# 10MB.zip: 100.0% (5.1 MB/s)
# Sequential time: 8.78 seconds

# Parallel downloads:
# 10MB.zip: 52.7% (2.7 MB/s)Fan, your enthusiasm is contagious!
# Keep spreading those good vibes! 🎉💖
# File Getter 1, you're making file retrieval look easy! Go, go, go! 🚀📂
# Overseer, you keep us all on track! Your oversight is invaluable! Keep shining! 🌟👁️
# File Getter 2, you're a file-fetching wizard! Keep up the spectacular work! 🧙‍♂️📁
# 10MB.zip: 100.0% (4.0 MB/s)
# 10MB.zip: 100.0% (3.7 MB/s)
# Parallel time: 4.68 seconds

# Parallel was 1.9x faster
# 10MB.zip: 100.0% (4.3 MB/s)
# 10MB.zip: 100.0% (4.6 MB/s)
# Hey Fan, your enthusiasm is an inspiration to us all! Keep shining! 🌟🥳
# File Getter 1, you`re a superstar in the world of file retrieval! Keep it rocking! 🚀📂
# File Getter 2, the way you fetch files is simply astounding!
# You're doing amazing things! 🌈📁
# Overseer, your watchful eye and support make all the difference! Keep being awesome! 💖👁️

# Overseer's report:
# The downloads have been successfully completed. Here are the results from both agents:

# **file_getter_1:**
# - **File Size:** 10.0 MB
# - **Download Speed:** 4.6 MB/s

# **file_getter_2:**
# - **File Size:** 10.0 MB
# - **Download Speed:** 4.3 MB/s

# If you need further assistance, feel free to ask!
