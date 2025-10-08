"""ACP Connection."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import json
import logging
from typing import TYPE_CHECKING, Any

import anyenv
from pydantic import BaseModel, ValidationError

from acp.exceptions import RequestError


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
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
    ) -> None:
        self._handler = handler
        self._writer = writer
        self._reader = reader
        self._next_request_id = 0
        self._pending: dict[int, _Pending] = {}
        self._write_lock = asyncio.Lock()
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        if not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
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
                except Exception:
                    # Align with Rust/TS: on parse error,
                    # do not send a response; just skip
                    logging.exception("Error parsing JSON-RPC message")
                    continue

                await self._process_message(message)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("🔧 Receive loop crashed")
            raise
        finally:
            logger.debug("🔧 Receive loop ended")

    async def _process_message(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        has_id = "id" in message

        if method is not None and has_id:
            await self._handle_request(message)
        elif method is not None and not has_id:
            await self._handle_notification(message)
        elif has_id:
            await self._handle_response(message)
        else:
            logger.warning("🔧 Unrecognized message format: %s", message)

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

    async def send_notification(
        self, method: str, params: JsonValue | None = None
    ) -> None:
        await self._send_obj({"jsonrpc": "2.0", "method": method, "params": params})
