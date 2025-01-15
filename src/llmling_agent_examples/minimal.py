"""Example: Different agents interpret the same facts based on their perspective."""

from llmling_agent import Agent


TASK = """
Fetch the Wikipedia article about climate change from https://en.wikipedia.org/wiki/Climate_change.
Use the URL as an argument.
"""

REPORTER_PROMPT = "Shortly search for climate change trends in 2024 and report.."
ECO_PROMPT = (
    "You're a passionate eco-warrior who sees imminent doom in every temperature rise!"
)
SKEPTIC_PROMPT = (
    "You're that uncle who thinks snowfall disproves global warming. Everything's fine!"
)
MODEL = "openai:gpt-4o-mini"


async def main():
    async with (
        Agent[None](
            model=MODEL,
            name="Reporter",
            system_prompt=REPORTER_PROMPT,
            mcp_servers=["uvx mcp-server-fetch"],
        ) as reporter,
        Agent[None](model=MODEL, name="EcoWarrior", system_prompt=ECO_PROMPT) as activist,
        Agent[None](model=MODEL, name="Skeptic", system_prompt=SKEPTIC_PROMPT) as skeptic,
    ):
        # Connect reporter to both interpreters and get Talk objects
        team_talk = reporter >> (activist | skeptic)
        activist.message_sent.connect(print)
        skeptic.message_sent.connect(print)
        # Send the news and monitor the connections
        result = await reporter.run(TASK)
    print(result)

    # Show what happened on the connections
    print("\nConnection Statistics:")
    print(f"Messages forwarded: {team_talk.stats.message_count}")
    print(f"Source: {team_talk.stats.source_names}")
    print(f"Targets: {team_talk.stats.target_names}")
    print(f"Total tokens: {team_talk.stats.token_count}")


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
