"""Example of documentation generation using agent delegation."""

from __future__ import annotations

import asyncio
import tempfile

from llmling_agent.delegation import AgentPool


AGENT_CONFIG = """
agents:
  overseer:
    name: "Documentation Overseer"
    model: openai:gpt-4o-mini
    system_prompts:
      - |
        You are a file analyzer. Your task is to:
        1. Get a list of all Python files in the given directory
        2. Create a clear summary of which files were found
    environment:
      type: inline
      tools:
        list_source_files:
          import_path: os.listdir

  doc_writer:
    name: "Documentation Writer"
    model: openai:gpt-4o-mini
    system_prompts:
      - |
        You are a technical documentation expert. Read all the source files
        and create a nice concise manual.md for them.
    environment:
      type: inline
      tools:
        read_source_file:
          # equals roughly to pathlib.Path.read_text
          import_path: llmling_agent_tools.file.read_source_file
"""


async def run(config_path: str):
    async with AgentPool.open(config_path) as pool:
        overseer = pool.get_agent("overseer")
        writer = pool.get_agent("doc_writer")

        # Connect overseer to writer
        overseer.pass_results_to(writer, prompt="Please write documentation for me.")
        # Connect writer output to our print function
        writer.outbox.connect(lambda msg, _: print(msg.content))

        # Just run the overseer - it will find files and pass them on
        await overseer.run('List all files in the "src/llmling_agent/agent" directory.')
        await writer.complete_tasks()  # everything is async, so we need to wait.


async def main():
    import logging

    logging.basicConfig(level=logging.DEBUG)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        await run(tmp.name)


if __name__ == "__main__":
    asyncio.run(main())
