import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.integrations.subprocess_runner import SubprocessRunner, WorkerTimeoutError


class FakeProcess:
    def __init__(self, *, completed=False, stdout=b"", stderr=b""):
        self.returncode = 0 if completed else None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr.feed_data(stderr)
        self.event = asyncio.Event()
        if completed:
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.event.set()
        self.killed = False
        self.wait_count = 0

    async def wait(self):
        self.wait_count += 1
        await self.event.wait()
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.event.set()


class SubprocessRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_invocation_is_shell_free_and_capture_is_bounded(self):
        process = FakeProcess(completed=True, stdout=b"a" * 100, stderr=b"b" * 100)
        create_process = AsyncMock(return_value=process)
        command = ["python.exe", "worker.py", "request.json", "result.json"]
        with patch("asyncio.create_subprocess_exec", create_process):
            result = await SubprocessRunner(capture_limit_bytes=16).run(
                command,
                cwd=Path.cwd(),
                timeout_seconds=1,
            )

        positional = create_process.await_args.args
        keywords = create_process.await_args.kwargs
        self.assertEqual(list(positional), command)
        self.assertNotIn("shell", keywords)
        self.assertEqual(len(result.stdout), 16)
        self.assertEqual(len(result.stderr), 16)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)

    async def test_timeout_kills_and_waits_for_process(self):
        process = FakeProcess()
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        ):
            with self.assertRaises(WorkerTimeoutError):
                await SubprocessRunner().run(
                    ["python.exe", "worker.py"],
                    cwd=Path.cwd(),
                    timeout_seconds=0.001,
                )
        self.assertTrue(process.killed)
        self.assertGreaterEqual(process.wait_count, 2)

    async def test_cancellation_kills_and_waits_for_process(self):
        process = FakeProcess()
        create_process = AsyncMock(return_value=process)
        with patch("asyncio.create_subprocess_exec", create_process):
            task = asyncio.create_task(
                SubprocessRunner().run(
                    ["python.exe", "worker.py"],
                    cwd=Path.cwd(),
                    timeout_seconds=30,
                )
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(create_process.await_count, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(process.killed)
        self.assertGreaterEqual(process.wait_count, 2)
