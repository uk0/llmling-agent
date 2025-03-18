"""Server for remote WebSocket provider."""

from __future__ import annotations

import asyncio

from fastapi import WebSocketDisconnect
import websockets

from llmling_agent.log import get_logger


logger = get_logger(__name__)


# Simple echo server for testing
async def handler(websocket):
    """Process WebSocket connection."""
    import anyenv

    await websocket.accept()
    try:
        while True:
            # Get raw message
            raw_message = await websocket.receive_text()
            message = anyenv.load_json(raw_message, return_type=dict)

            match message["type"]:
                case "init":
                    await websocket.send_json({
                        "type": "response",
                        "content": "Context received",
                        "ref_id": message.get("message_id"),
                    })

                case "prompt":
                    prompts = message["content"]["prompts"]
                    # Simulate tool usage
                    await websocket.send_json({
                        "type": "tool_call",
                        "content": {
                            "name": "echo_tool",
                            "args": {"text": prompts[0]},
                        },
                        "message_id": "test-tool-call",
                    })
                    # Wait for tool result
                    _tool_result = await websocket.receive_text()
                    # Send final response
                    await websocket.send_json({
                        "type": "response",
                        "content": {
                            "content": f"Echo: {prompts}",
                            "model": "echo-model",
                        },
                        "ref_id": message.get("message_id"),
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


async def main():
    async with websockets.serve(handler, "localhost", 8000):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
