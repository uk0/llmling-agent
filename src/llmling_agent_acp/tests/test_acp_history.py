"""Tests to verify ACP message history functionality."""

import asyncio
import tempfile

from slashed import CommandStore

from llmling_agent import Agent
from llmling_agent_acp import DefaultACPClient
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.converters import to_content_blocks
from llmling_agent_acp.session import ACPSessionManager


async def test_conversation_history():
    """Test that conversation history is maintained across multiple prompts."""
    print("🧪 Testing ACP conversation history...")

    # Create a simple agent
    agent = Agent(
        name="test_agent",
        model="openrouter:openai/gpt-5-mini",
        system_prompt="You are a helpful assistant. Remember our conversation.",
    )

    # Create session manager directly
    client = DefaultACPClient(allow_file_operations=False)

    command_store = CommandStore()
    command_bridge = ACPCommandBridge(command_store)

    session_manager = ACPSessionManager(command_bridge=command_bridge)

    # Create a session
    with tempfile.TemporaryDirectory() as temp_dir:
        session_id = await session_manager.create_session(
            agent=agent,
            cwd=temp_dir,
            client=client,
        )
        session = await session_manager.get_session(session_id)
        assert session

        print(f"✅ Created session: {session.session_id}")

        # Test 1: Send first message
        print("\n📤 Sending first message: 'My favorite color is blue'")
        content_blocks_1 = to_content_blocks("My favorite color is blue")

        # Process the prompt and collect responses
        responses_1 = []
        async for result in session.process_prompt(content_blocks_1):
            if isinstance(result, str):
                print(f"   Stop reason: {result}")
                break
            responses_1.append(result)

        print(f"   Got {len(responses_1)} response chunks")

        # Test 2: Send second message that references the first
        print("\n📤 Sending second message: 'What color did I just mention?'")
        content_blocks_2 = to_content_blocks("What color did I just mention?")

        # Process the second prompt
        responses_2 = []
        async for result in session.process_prompt(content_blocks_2):
            if isinstance(result, str):
                print(f"   Stop reason: {result}")
                break
            responses_2.append(result)
            # Print the actual response content for verification
            if hasattr(result.update, "content"):
                content = result.update.content  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(content, "text"):
                    print(f"   Response chunk: {content.text}")  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        print(f"   Got {len(responses_2)} response chunks")

        agent_history = session.agent.conversation.get_history()
        print(f"\n🧠 Agent's internal history ({len(agent_history)} messages):")
        for i, msg in enumerate(agent_history):
            role_icon = "🧑" if msg.role == "user" else "🤖"
            print(f"   [{i}] {role_icon} {msg.role}: {msg.content}")

        # Verify that the agent has the full conversation context
        if len(agent_history) >= 2:  # noqa: PLR2004
            first_user_msg = next(
                (msg for msg in agent_history if msg.role == "user"), None
            )
            if first_user_msg and "blue" in first_user_msg.content.lower():
                print("\n✅ SUCCESS: Agent has access to first message about blue color")
            else:
                print("\n❌ FAILURE: Agent missing first message about blue color")
        else:
            print(f"\n❌ FAILURE: Agent only has {len(agent_history)} msgs in history")

        # Close the session
        await session.close()
        print("\n🏁 Test completed")


async def test_simple_sync():
    """Simple test of agent conversation history baseline."""
    print("\n🔧 Testing agent conversation baseline...")

    # Create a simple agent
    agent = Agent(name="sync_test_agent", model="openrouter:openai/gpt-5-mini")

    client = DefaultACPClient()

    command_store = CommandStore()
    command_bridge = ACPCommandBridge(command_store)

    session_manager = ACPSessionManager(command_bridge=command_bridge)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = await session_manager.create_session(
                agent=agent,
                cwd=temp_dir,
                client=client,
            )
            session = await session_manager.get_session(session_id)
            assert session

            # Check agent's history (should be empty since no prompts were processed)
            agent_history = session.agent.conversation.get_history()
            print(f"Agent history: {len(agent_history)} messages")

            for msg in agent_history:
                print(f"  {msg.role}: {msg.content}")

            if len(agent_history) == 0:
                print("✅ SUCCESS: Agent starts with empty history as expected")
            else:
                print("❌ FAILURE: Agent should start with empty history")

            await session.close()

    except Exception as e:  # noqa: BLE001
        print(f"💥 Baseline test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting ACP conversation history tests...")
    asyncio.run(test_simple_sync())
    asyncio.run(test_conversation_history())
