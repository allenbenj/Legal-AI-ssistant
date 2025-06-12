import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from legal_ai_system.services.automatic_processing import AutomaticProcessingPipeline


class DummyIntegration:
    def __init__(self) -> None:
        self.calls = []

    async def upload_and_process_document(self, content: bytes, filename: str, user, options):
        self.calls.append(filename)
        return {"document_id": filename}


class AutomaticProcessingPipelineTests(unittest.TestCase):
    def test_priority_order(self) -> None:
        with TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "low.txt"
            p2 = Path(tmp) / "high.txt"
            p1.write_text("a")
            p2.write_text("b")

            integration = DummyIntegration()
            pipeline = AutomaticProcessingPipeline(integration, [], max_concurrent=1)

            async def run() -> None:
                pipeline._running = True
                worker = asyncio.create_task(pipeline._process_queue())
                await pipeline.enqueue_file(p1, priority=5)
                await pipeline.enqueue_file(p2, priority=1)
                await asyncio.sleep(0.1)
                pipeline._running = False
                await pipeline._queue.join()
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

            asyncio.run(run())
            self.assertEqual(integration.calls, ["high.txt", "low.txt"])


if __name__ == "__main__":
    unittest.main()
