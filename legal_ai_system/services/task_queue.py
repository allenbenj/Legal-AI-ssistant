from __future__ import annotations

"""Simple wrapper around RQ for task queuing."""

try:
    from rq import Queue
    from redis import Redis
except Exception:  # pragma: no cover - dependency may be missing during tests
    Queue = None  # type: ignore
    Redis = None  # type: ignore


class TaskQueue:
    """Task queue based on Redis Queue (RQ)."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        if Redis is None or Queue is None:
            self.connection = None
            self.queue = None
        else:
            self.connection = Redis.from_url(redis_url)
            self.queue = Queue(connection=self.connection)

    def enqueue(self, func, *args, **kwargs):
        """Enqueue a task for asynchronous execution."""
        if self.queue is None:
            raise RuntimeError("Task queue backend not available")
        return self.queue.enqueue(func, *args, **kwargs)

