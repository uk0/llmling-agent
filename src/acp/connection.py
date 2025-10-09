"""ACP Connection."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyenv
from pydantic import BaseModel, ValidationError

from acp.exceptions import RequestError
from llmling_agent import log


logger = log.get_logger(__name__)


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from acp.acp_types import JsonValue, MethodHandler


@dataclass(slots=True)
class _Pending:
    future: asyncio.Future[Any]


class Connection:
    """Minimal JSON-RPC 2.0 connection over newline-delimited JSON frames.

    Using asyncio streams. KISS: only supports StreamReader/StreamWriter.

    - Outgoing messages always include {"jsonrpc": "2.0"}
    - Requests and notifications are dispatched to a single async handler
    - Responses resolve pending futures by numeric id
    """

    def __init__(
        self,
        handler: MethodHandler,
        writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader,
        debug_messages: bool = False,
        debug_file: str | None = None,
    ) -> None:
        self._handler = handler
        self._writer = writer
        self._reader = reader
        self._debug_messages = debug_messages
        self._debug_file = Path(debug_file) if debug_file else None
        self._next_request_id = 0
        self._pending: dict[int, _Pending] = {}
        self._inflight: set[asyncio.Task[Any]] = set()
        self._write_lock = asyncio.Lock()
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        if not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
        if self._inflight:
            tasks = list(self._inflight)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        # Do not close writer here; lifecycle owned by caller

    # --- IO loops ----------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    message = anyenv.load_json(line, return_type=dict)
                    if self._debug_messages and self._debug_file:
                        self._write_debug_message("←", line.decode().strip())
                except Exception:
                    # Align with Rust/TS: on parse error,
                    # do not send a response; just skip
                    logger.exception("Error parsing JSON-RPC message")
                    continue

                await self._process_message(message)
        except asyncio.CancelledError:
            return

    async def _process_message(self, message: dict) -> None:
        method = message.get("method")
        has_id = "id" in message

        if method is not None and has_id:
            self._schedule(self._handle_request(message))
            return
        if method is not None and not has_id:
            await self._handle_notification(message)
            return
        if has_id:
            await self._handle_response(message)

    def _schedule(self, coro: Coroutine[None, None, Any]) -> None:
        task = asyncio.create_task(coro)
        self._inflight.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._inflight.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Unhandled error in JSON-RPC request handler")

    async def _handle_request(self, message: dict[str, Any]) -> None:
        """Handle JSON-RPC request."""
        payload = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            result = await self._handler(message["method"], message.get("params"), False)
            if isinstance(result, BaseModel):
                result = result.model_dump(by_alias=True, exclude_none=True)
            payload["result"] = result if result is not None else None
        except RequestError as re:
            payload["error"] = re.to_error_obj()
        except ValidationError as ve:
            data: dict[str, Any] = {"errors": ve.errors()}
            payload["error"] = RequestError.invalid_params(data).to_error_obj()
        except Exception as err:  # noqa: BLE001
            try:
                data = anyenv.load_json(str(err), return_type=dict)
            except Exception:  # noqa: BLE001
                data = {"details": str(err)}
            payload["error"] = RequestError.internal_error(data).to_error_obj()
        await self._send_obj(payload)

    async def _handle_notification(self, message: dict[str, Any]) -> None:
        """Handle JSON-RPC notification."""
        with contextlib.suppress(Exception):
            # Best-effort; notifications do not produce responses
            await self._handler(message["method"], message.get("params"), True)

    async def _handle_response(self, message: dict[str, Any]) -> None:
        """Handle JSON-RPC response."""
        fut = self._pending.pop(message["id"], None)
        if fut is None:
            return
        if "result" in message:
            fut.future.set_result(message.get("result"))
        elif "error" in message:
            err = message.get("error") or {}
            code = err.get("code", -32603)
            error = RequestError(code, err.get("message", "Error"), err.get("data"))
            fut.future.set_exception(error)
        else:
            fut.future.set_result(None)

    async def _send_obj(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        if self._debug_messages and self._debug_file:
            self._write_debug_message("→", json.dumps(obj, separators=(",", ":")))
        async with self._write_lock:
            self._writer.write(data)
            with contextlib.suppress(ConnectionError, RuntimeError):
                # Peer closed; let reader loop end naturally
                await self._writer.drain()

    # --- Public API --------------------------------------------------------------

    async def send_request(
        self, method: str, params: JsonValue | None = None
    ) -> dict[str, Any]:
        req_id = self._next_request_id
        self._next_request_id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = _Pending(fut)
        dct = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        await self._send_obj(dct)
        return await fut

    def _write_debug_message(self, direction: str, message: str) -> None:
        """Write debug message to file."""
        if not self._debug_file:
            return
        try:
            timestamp = datetime.datetime.now().isoformat()
            debug_line = f"{timestamp} {direction} {message}\n"
            with self._debug_file.open("a", encoding="utf-8") as f:
                f.write(debug_line)
        except Exception:  # noqa: BLE001
            # Don't let debug logging break the connection
            pass

    async def send_notification(
        self, method: str, params: JsonValue | None = None
    ) -> None:
        await self._send_obj({"jsonrpc": "2.0", "method": method, "params": params})
