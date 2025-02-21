# llmling-agent\src\llmling_agent_examples\round_robin\main.py
# /// script
# dependencies = ["llmling-agent"]
# ///

"""Run round-robin example demonstrating cyclic communication pattern."""

from __future__ import annotations

from llmling_agent.__main__ import run_command
from llmling_agent_examples.utils import get_config_path, is_pyodide


if __name__ == "__main__":
    # Use utils to get config path that works in both environments
    config_path = get_config_path(None if is_pyodide() else __file__)

    run_command(
        node_name="player1",
        prompts=["Start the word chain with: tree"],
        config_path=str(config_path),
        show_messages=True,
        detail_level="simple",
        show_metadata=False,
        show_costs=False,
        verbose=False,
    )
