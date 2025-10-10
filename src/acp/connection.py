"""ACP Connection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Self

import anyenv
from pydantic import BaseModel, ValidationError

from acp.exceptions import RequestError
from acp.task import (
    DefaultMessageDispatcher,
    InMemoryMessageQueue,
    InMemoryMessageStateStore,
    MessageDispatcher,
    MessageQueue,
    MessageSender,
    MessageStateStore,
    NotificationRunner,
    RequestRunner,
    RpcTask,
    RpcTaskKind,
    TaskSupervisor,
)
from llmling_agent import log


logger = log.get_logger(__name__)

if TYPE_CHECKING:
    from acp.acp_types import JsonValue, MethodHandler
    from acp.task import SenderFactory

__all__ = ["Connection"]


DispatcherFactory = Callable[
    [MessageQueue, TaskSupervisor, MessageStateStore, RequestRunner, NotificationRunner],
    MessageDispatcher,
]


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
        *,
        queue: MessageQueue | None = None,
        state_store: MessageStateStore | None = None,
        dispatcher_factory: DispatcherFactory | None = None,
        sender_factory: SenderFactory | None = None,
    ) -> None:
        self._handler = handler
        self._writer = writer
        self._reader = reader
        self._next_request_id = 0
        self._state = state_store or InMemoryMessageStateStore()
        self._tasks = TaskSupervisor(source="acp.Connection")
        self._tasks.add_error_handler(self._on_task_error)
        self._queue = queue or InMemoryMessageQueue()
        self._closed = False
        self._sender = (sender_factory or MessageSender)(self._writer, self._tasks)
        self._recv_task = self._tasks.create(
            self._receive_loop(),
            name="acp.Connection.receive",
            on_error=self._on_receive_error,
        )
        dispatcher_factory = dispatcher_factory or self._default_dispatcher_factory
        self._dispatcher = dispatcher_factory(
            self._queue,
            self._tasks,
            self._state,
            self._run_request,
            self._run_notification,
        )
        self._dispatcher.start()

    async def close(self) -> None:
        """Stop the receive loop and cancel any in-flight handler tasks."""
        if self._closed:
            return
        self._closed = True
        await self._dispatcher.stop()
        await self._sender.close()
        await self._tasks.shutdown()
        self._state.reject_all_outgoing(ConnectionError("Connection closed"))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def send_request(self, method: str, params: JsonValue | None = None) -> Any:
        request_id = self._next_request_id
        self._next_request_id += 1
        future = self._state.register_outgoing(request_id, method)
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        await self._sender.send(payload)
        return await future

    async def send_notification(
        self, method: str, params: JsonValue | None = None
    ) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._sender.send(payload)

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
                    logger.exception("Error parsing JSON-RPC message")
                    continue

                await self._process_message(message)
        except asyncio.CancelledError:
            return

    async def _process_message(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        has_id = "id" in message
        if method is not None and has_id:
            await self._queue.publish(RpcTask(RpcTaskKind.REQUEST, message))
            return
        if method is not None and not has_id:
            await self._queue.publish(RpcTask(RpcTaskKind.NOTIFICATION, message))
            return
        if has_id:
            await self._handle_response(message)

    async def _send_obj(self, obj: dict[str, Any]) -> None:
        await self._sender.send(obj)

    async def _run_request(self, message: dict[str, Any]) -> Any:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            result = await self._handler(message["method"], message.get("params"), False)
            if isinstance(result, BaseModel):
                result = result.model_dump(by_alias=True, exclude_none=True)
            payload["result"] = result if result is not None else None
            await self._sender.send(payload)
            return payload.get("result")
        except RequestError as exc:
            payload["error"] = exc.to_error_obj()
            await self._sender.send(payload)
            raise
        except ValidationError as exc:
            err = RequestError.invalid_params({"errors": exc.errors()})
            payload["error"] = err.to_error_obj()
            await self._sender.send(payload)
            raise err from None
        except Exception as exc:  # noqa: BLE001
            try:
                data = anyenv.load_json(str(exc), return_type=dict)
            except Exception:  # noqa: BLE001
                data = {"details": str(exc)}
            err = RequestError.internal_error(data)
            payload["error"] = err.to_error_obj()
            await self._sender.send(payload)
            raise err from None

    async def _run_notification(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await self._handler(message["method"], message.get("params"), True)

    async def _handle_response(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        result = message.get("result")
        if "result" in message:
            self._state.resolve_outgoing(request_id, result)
            return
        if "error" in message:
            error_obj = message.get("error") or {}
            self._state.reject_outgoing(
                request_id,
                RequestError(
                    error_obj.get("code", -32603),
                    error_obj.get("message", "Error"),
                    error_obj.get("data"),
                ),
            )
            return
        self._state.resolve_outgoing(request_id, None)

    def _on_receive_error(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        logging.exception("Receive loop failed", exc_info=exc)
        self._state.reject_all_outgoing(exc)

    def _on_task_error(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        logging.exception("Background task failed", exc_info=exc)

    def _default_dispatcher_factory(
        self,
        queue: MessageQueue,
        supervisor: TaskSupervisor,
        state: MessageStateStore,
        request_runner: RequestRunner,
        notification_runner: NotificationRunner,
    ) -> MessageDispatcher:
        return DefaultMessageDispatcher(
            queue=queue,
            supervisor=supervisor,
            store=state,
            request_runner=request_runner,
            notification_runner=notification_runner,
        )
