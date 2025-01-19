# Multi-Agent Documentation System

This example demonstrates a team of three agents working together to scan, document, and validate Python code. It shows different patterns of agent collaboration:

- Async delegation (fire and forget)
- Tool usage with waiting
- Chained tool calls

## Configuration

First, let's set up our agents in `docs_agents.yml`:

```yaml
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
```

## Implementation

Here's how we orchestrate these agents:

```python
from mypy import api
from llmling_agent.delegation import AgentPool

def check_types(path: str) -> str:
    """Type check Python file using mypy."""
    stdout, _stderr, _code = api.run([path])
    return stdout

async def main():
    # Initialize our agent pool with our configuration
    async with AgentPool[None]("docs_agents.yml") as pool:
        scanner = pool.get_agent("file_scanner")
        writer = pool.get_agent("doc_writer")
        checker = pool.get_agent("error_checker")

        # Setup async chain: scanner -> writer -> console output
        scanner >> writer
        writer.message_sent.connect(print)

        # Start async docs generation
        await scanner.run('List all Python files in "src/llmling_agent/agent"')

        # Use error checker as tool (this blocks until complete)
        scanner.register_worker(checker)

        prompt = 'Check types for all Python files in "src/llmling_agent/agent"'
        result = await scanner.run(prompt)
        print(f"Type checking result:\n{result.data}")

        # Wait for async documentation to finish
        await writer.complete_tasks()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## How It Works

1. The File Scanner agent scans directories and identifies Python files
2. It passes these files to the Documentation Writer agent asynchronously
3. In parallel, it uses the Error Checker as a tool to validate the files
4. The Documentation Writer processes files as they come in
5. Results are printed to the console as they become available

This demonstrates different ways agents can collaborate:

- Async message passing (scanner to writer)
- Synchronous tool usage (scanner using checker)
- Event-based output handling (writer to console)
