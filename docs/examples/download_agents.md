# Multi-Agent Download System with Cheerleader

This example demonstrates several advanced features of LLMling-agent:

- Continuous repetitive tasks
- Async parallel execution of LLM calls
- YAML configuration with storage providers
- Capability usage (agent listing and task delegation)
- Stateful callback mechanism
- Multiple storage providers (SQLite + pretty-printed logs)

## Configuration

Our `download_agents.yml` defines four agents and storage configuration:

```yaml
storage:
  # List of storage providers (can use multiple)
  providers:
    # Primary storage using SQLite
    - type: sql
      url: "sqlite:///history.db"
    # Pretty printed text file output
    - type: text_file
      path: "logs/chat.log"
      format: "chronological"
      template: "chronological"

agents:
  fan:
    name: "Async Agent Fan"
    description: "The #1 supporter of all agents!"
    model:
      type: fallback
      models:
          - openai:gpt-4o-mini
          - openai:gpt-3.5-turbo
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
```

## Implementation

Here's how we orchestrate our download team:

```python
class CheerProgress:
    """State keeper for our enthusiastic fan."""
    def __init__(self):
        self.situation = "The team is assembling, ready to start the downloads!"

    def create_prompt(self) -> str:
        return (
            f"Current situation: {self.situation}\n"
            "Be an enthusiastic and encouraging fan!"
        )

    def update(self, situation: str):
        self.situation = situation
        print(situation)

async def run():
    async with AgentPool[None]("download_agents.yml") as pool:
        # Get first worker from config
        worker_1 = pool.get_agent("file_getter_1")

        # Clone it for second worker
        worker_2 = await pool.clone_agent(worker_1, new_name="file_getter_2")

        # Create team and get fan
        team = worker_1 & worker_2
        fan = pool.get_agent("fan")
        progress = CheerProgress()

        # Start continuous fan support in background
        await fan.run_continuous(progress.create_prompt)

        # Sequential downloads
        progress.update("Sequential downloads starting - let's see how they do!")
        sequential = await team.run_sequential(TEAM_PROMPT)
        progress.update(f"Downloads completed in {sequential.duration:.2f} secs!")

        # Parallel downloads
        parallel = await team.run_parallel(TEAM_PROMPT)
        progress.update(f"Downloads completed in {parallel.duration:.2f} secs!")

        # Let overseer coordinate
        overseer = pool.get_agent("overseer")
        result = await overseer.run(OVERSEER_PROMPT)
        progress.update(f"\nOverseer's report: {result.data}")

        await fan.stop()  # End of joy

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

## How It Works

1. We set up a team of downloaders and a cheerleading fan
2. The fan runs continuously in the background, getting updates via callbacks
3. We test downloads in different modes:
   - Sequential (one after another)
   - Parallel (both at once)
   - Overseer-coordinated (using agent capabilities)
4. The fan cheers appropriately for each situation
5. All interactions are logged to both SQLite and pretty-printed text files

This demonstrates:

- Background tasks with continuous prompts
- Agent cloning
- Team operations (sequential vs parallel)
- Capability-based delegation
- Multi-provider storage
