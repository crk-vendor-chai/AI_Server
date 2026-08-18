"""Thread-safe per-division queue for serial annotation request batches."""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple


class SerialBatchQueue:
    """Keep one drain worker per key while allowing requests to append batches.

    Each worker owns its key through a token. Ownership lets a worker that dies
    unexpectedly release the key without stealing it from its successor, so a
    later request can always start a fresh worker instead of queueing forever.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[Any]] = defaultdict(deque)
        self._active: Dict[str, int] = {}
        self._next_token = 0

    def enqueue(self, key: str, item: Any) -> Tuple[Optional[int], int]:
        """Append an item and return (worker_token, queued_count).

        worker_token is None when a worker already owns the key; otherwise the
        caller must start a drain worker with that token.
        """
        with self._lock:
            queue = self._queues[key]
            queue.append(item)
            if key in self._active:
                return None, len(queue)
            self._next_token += 1
            self._active[key] = self._next_token
            return self._next_token, len(queue)

    def pop(self, key: str, token: int) -> Optional[Any]:
        """Pop the next batch; atomically retire the worker when the queue is empty."""
        with self._lock:
            if self._active.get(key) != token:
                return None
            queue = self._queues.get(key)
            if queue:
                return queue.popleft()
            self._queues.pop(key, None)
            self._active.pop(key, None)
            return None

    def retire(self, key: str, token: int) -> bool:
        """Release the key if this worker still owns it. Safe to call twice."""
        with self._lock:
            if self._active.get(key) != token:
                return False
            self._active.pop(key, None)
            return True

    def pending_count(self, key: str) -> int:
        with self._lock:
            return len(self._queues.get(key) or ())

    def clear(self) -> None:
        """Reset queue state. Intended for isolated tests."""
        with self._lock:
            self._queues.clear()
            self._active.clear()


annotation_batch_queue = SerialBatchQueue()
