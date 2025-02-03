"""Logging configuration for llmling_agent."""

from __future__ import annotations

from io import StringIO
import logging
from queue import Queue
import threading
import time


class LogCapturer:
    """Captures log output for display in UI."""

    def __init__(self):
        """Initialize log capturer."""
        self.log_queue: Queue[str] = Queue()
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(fmt)

    def start(self):
        """Start capturing logs."""
        logging.getLogger().addHandler(self.handler)

        def monitor():
            while True:
                if self.buffer.tell():
                    self.buffer.seek(0)
                    self.log_queue.put(self.buffer.getvalue())
                    self.buffer.truncate(0)
                    self.buffer.seek(0)
                time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop capturing logs."""
        logging.getLogger().removeHandler(self.handler)

    def get_logs(self) -> str:
        """Get accumulated logs."""
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get_nowait())
        return "".join(logs)
