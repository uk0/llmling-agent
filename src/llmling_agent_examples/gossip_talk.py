"""Office Gossip: A demonstration of agent communication patterns.

This example shows:
1. Direct message without response (Alice -> Bob)
2. Message with expected response (Carol -> Bob)
3. Type-safe agent communication
"""

from __future__ import annotations

from llmling_agent.delegation import AgentPool


AGENT_CONFIG = """
agents:
  alice:
    name: "Alice from Accounting"
    model: openai:gpt-4o-mini
    system_prompts:
      - You are Alice, known for sharing office gossip. You love talking about coworkers.
    knowledge:
        resources:
            - type: text
            - text: |
             Hey Bob, did you hear? I saw Dave from Engineering walking into
             the CEO's office yesterday with a huge smile! And this morning
             he was cleaning out his desk! 👀
  bob:
    name: "Bob from HR"
    model: openai:gpt-4o-mini
    system_prompts:
      - |
        You are Bob from HR. You're diplomatic and remember everything people tell you.
        When asked about what others said, you relay the information professionally.

  carol:
    name: "Carol from Marketing"
    model: openai:gpt-4o-mini
    system_prompts:
      - You are Carol from Marketing. You're always curious about office happenings.
"""


async def main(config_path: str):
    async with AgentPool[None](config_path) as pool:
        alice = pool.get_agent("alice")
        bob = pool.get_agent("bob")
        carol = pool.get_agent("carol")

        # Alice tells Bob about Dave's "promotion"
        # This just appends a message to Bob's conversation history
        await alice.share(bob, resources=["resource_name"])

        # Carol asks Bob what Alice said (expects response)
        response = await carol.talk.ask(
            bob,
            "Hey Bob, I saw Alice talking to you earlier. "
            "Any interesting office updates? 😊",
        )
        print(f"\nCarol got the gossip:\n{response.data}")


if __name__ == "__main__":
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        asyncio.run(main(tmp.name))
