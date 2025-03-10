import pytest

from llmling_agent import Agent
from llmling_agent.models.content import ImageURLContent


@pytest.mark.asyncio
async def test_litellm_vision():
    """Test basic vision capability with a small, public image."""
    agent = Agent[None](provider="pydantic_ai", name="test-vision", model="gpt-4o")

    # Using a small, public image
    msg = "https://python.org/static/community_logos/python-logo-master-v3-TM.png"
    image = ImageURLContent(url=msg, description="Python logo")
    msg = "What does this image show? Answer in one short sentence."
    result = await agent.run(msg, image)

    assert isinstance(result.content, str)
    assert "Python" in result.content
    assert len(result.content) < 100  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main([__file__])
