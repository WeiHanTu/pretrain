"""Per-rank structured event log.

One JSONL file per rank.  Ranks never share a file handle: interleaved writes from
several processes corrupt lines, and the corruption shows up exactly when the log
matters most, during a failure.

Every record carries ``rank`` and ``host`` so a straggler or a single misbehaving
node can be isolated from the merged stream after the fact.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = ["EventLog"]


class EventLog:
    """Append-only JSONL writer for one rank."""

    def __init__(self, path: Path, *, run_id: str, rank: int = 0) -> None:
        self.path = path
        self.run_id = run_id
        self.rank = rank
        self.host = socket.gethostname()
        self._pid = os.getpid()
        self._t0 = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def emit(self, event: str, /, **payload: Any) -> None:
        record = {
            "run_id": self.run_id,
            "t": round(time.monotonic() - self._t0, 6),
            "wall": time.time(),
            "rank": self.rank,
            "host": self.host,
            "pid": self._pid,
            "event": event,
            **payload,
        }
        self._fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.emit(
                "run_failed", error_type=exc_type.__name__ if exc_type else None, error=str(exc)
            )
        self.close()
