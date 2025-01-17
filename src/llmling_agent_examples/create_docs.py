"""Agentsoft Corp. 3 agents publishing software.

This example shows:
1. Async delegation: File scanner delegates to doc writer (fire and forget)
2. Tool usage (async + wait): File scanner uses error checker as a tool (wait for result)
3. Chained tool calls.
"""

from __future__ import annotations

from mypy import api

from llmling_agent.delegation import AgentPool


def check_types(path: str) -> str:
    """Type check Python file using mypy."""
    stdout, _stderr, _code = api.run([path])
    return stdout


AGENT_CONFIG = """
agents:
  file_scanner:
    name: "File Scanner"
    model: openai:gpt-4o-mini
    system_prompts:
      - You scan directories and list source files that need documentation.
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
          import_path: llmling_agent_tools.file.read_source_file

  error_checker:
    name: "Code Validator"
    model: openai:gpt-4o-mini
    system_prompts:
      - You validate Python source files for syntax errors.
    environment:
      type: inline
      tools:
        validate_syntax:
          import_path: __main__.check_types
          description: Type check Python file using mypy.
"""


async def main(config_path: str):
    async with AgentPool[None](config_path) as pool:
        scanner = pool.get_agent("file_scanner")
        writer = pool.get_agent("doc_writer")
        checker = pool.get_agent("error_checker")

        # Setup async chain: scanner -> writer -> console output
        scanner.pass_results_to(writer)
        writer.message_sent.connect(lambda msg: print(f"Documentation:\n{msg.content}"))
        # Start async docs generation (the writer will start working in async fashion)
        await scanner.run('List all Python files in "src/llmling_agent/agent"')

        # Use error checker as tool (this blocks until complete)
        scanner.register_worker(checker)
        prompt = 'Check types for all Python files in "src/llmling_agent/agent"'
        result = await scanner.run(prompt)
        print(f"Type checking result:\n{result.data}")

        # Wait for async documentation to finish and print.
        await writer.complete_tasks()


if __name__ == "__main__":
    import asyncio
    import logging
    import tempfile

    logging.basicConfig(level=logging.DEBUG)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
