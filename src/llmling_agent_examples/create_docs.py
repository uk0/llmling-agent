"""Shows how to chain two agents together for documentation generation.

The first agent lists files in a directory, then passes the list to a second agent
that reads the files and generates markdown documentation for them.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile

from llmling_agent.delegation import AgentPool


AGENT_CONFIG = """
agents:
  file_lister:
    name: "File lister"
    model: openai:gpt-4o-mini
    system_prompts:
      - You are a file-lister. List all files from the folder given to you.
    environment:
      type: inline
      tools:
        list_source_files:
          import_path: os.listdir

  doc_writer:
    name: "Documentation Writer"
    model: openai:gpt-4o-mini
    system_prompts:
      - You are a docs writer. Write markdown documentation for the files given to you.
    environment:
      type: inline
      tools:
        read_source_file:
          # equals roughly to pathlib.Path.read_text
          import_path: llmling_agent_tools.file.read_source_file
"""


async def main(config_path: str):
    async with AgentPool.open(config_path) as pool:
        file_lister = pool.get_agent("file_lister")
        writer = pool.get_agent("doc_writer")

        # Connect file_lister to writer
        file_lister.pass_results_to(writer, prompt="Please write documentation for me.")
        # Connect writer output to our print function
        writer.outbox.connect(lambda msg, _: print(msg.content))

        # Just run the file_lister - it will find files and pass them on
        await file_lister.run(
            'List all files in the "src/llmling_agent/agent" directory.'
        )
        await writer.complete_tasks()  # everything is async, so we need to wait.


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml") as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
