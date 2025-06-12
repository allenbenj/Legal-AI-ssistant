from __future__ import annotations

"""Automatic processing pipeline for uploaded documents.

This service monitors directories for new files and enqueues them for
processing via :class:`LegalAIIntegrationService`.
It supports configurable agent selection, task prioritisation, and
basic batch processing using an ``asyncio.PriorityQueue``.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from watchgod import Change, awatch


class AutomaticProcessingPipeline:
    """Automatically process files as they appear in watched directories."""

    def __init__(
        self,
        integration_service: Any,
        watch_directories: Iterable[Path],
        enabled_agents: Optional[List[str]] | None = None,
        max_concurrent: int = 2,
    ) -> None:
        self.integration_service = integration_service
        self.watch_directories = [Path(d) for d in watch_directories]
        self.enabled_agents = enabled_agents or []
        self.max_concurrent = max_concurrent
        self._queue: asyncio.PriorityQueue[
            Tuple[int, Tuple[Path, Optional[Any], Dict[str, Any]]]
        ] = asyncio.PriorityQueue()
        self._running = False
        self._workers: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start watching directories and processing the queue."""
        if self._running:
            return
        self._running = True
        for directory in self.watch_directories:
            self._workers.append(asyncio.create_task(self._watch_directory(directory)))
        self._workers.append(asyncio.create_task(self._process_queue()))

    async def stop(self) -> None:
        """Stop the pipeline and cancel running tasks."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def enqueue_file(
        self,
        path: Path,
        priority: int = 10,
        user: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a file to the processing queue."""
        await self._queue.put((priority, (path, user, options or {})))

    async def _watch_directory(self, directory: Path) -> None:
        async for changes in awatch(directory):
            if not self._running:
                break
            for change, filepath in changes:
                if change is Change.added:
                    await self.enqueue_file(Path(filepath))

    async def _process_queue(self) -> None:
        sem = asyncio.Semaphore(self.max_concurrent)
        while self._running:
            priority, (path, user, options) = await self._queue.get()
            if not path.exists():
                self._queue.task_done()
                continue
            async with sem:
                await self._process_file(path, user, options)
            self._queue.task_done()

    async def _process_file(
        self, path: Path, user: Optional[Any], options: Dict[str, Any]
    ) -> None:
        with open(path, "rb") as f:
            data = f.read()
        opts = dict(options)
        if self.enabled_agents:
            opts["enabled_agents"] = self.enabled_agents
        await self.integration_service.upload_and_process_document(
            data, path.name, user or getattr(self.integration_service, "default_user", None), opts
        )

__all__ = ["AutomaticProcessingPipeline"]
