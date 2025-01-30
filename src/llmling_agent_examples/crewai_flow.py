from __future__ import annotations

from llmling_agent import Agent  # noqa: TC001
from llmling_agent.models.agents import AgentsManifest
from llmling_agent.running import agent_function, run_agents_async


agents_yaml = """
agents:
  city_picker:
    model: gpt-4o-mini
    system_prompts: ["You generate random city names."]
  fact_finder:
    model: gpt-4o-mini
    system_prompts: ["You provide interesting facts about cities."]
"""


@agent_function
async def generate_city(city_picker: Agent[None]) -> str:
    """Generate a random city name."""
    result = await city_picker.run("Return the name of a random city in the world.")
    return result.data


@agent_function(depends_on=generate_city)
async def generate_fun_fact(fact_finder: Agent[None], generate_city: str) -> str:
    """Generate fun fact about the city."""
    result = await fact_finder.run(f"Tell me a fun fact about {generate_city}")
    return result.data


# Execute the flow
async def main():
    manifest = AgentsManifest.from_yaml(agents_yaml)
    results = await run_agents_async(manifest)
    print(f"City: {results['generate_city']}")
    print(f"Fun fact: {results['generate_fun_fact']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
