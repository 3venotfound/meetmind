import asyncio
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


DEFAULT_CAPTURE_LIMIT_BYTES = 65_536


class WorkerTimeoutError(TimeoutError):
    pass


class WorkerLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool


class WorkerRunner(Protocol):
    async def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        extra_environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        ...


def resolve_python_executable(configured_value: str) -> Path:
    raw_value = configured_value.strip()
    candidate = Path(raw_value).expanduser() if raw_value else Path(sys.executable)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkerLaunchError("Python executable is unavailable") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise WorkerLaunchError("Python executable is unavailable")
    return resolved


async def _read_limited(
    stream: asyncio.StreamReader | None,
    limit_bytes: int,
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    captured = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = limit_bytes - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(captured), truncated


async def _kill_and_wait(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


class _ThreadedProcess:
    """Run a child process without relying on event-loop subprocess support."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: float,
        capture_limit_bytes: int,
    ):
        self.command = command
        self.cwd = cwd
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.capture_limit_bytes = capture_limit_bytes
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def run(self) -> ProcessResult:
        try:
            process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                env=dict(self.environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except (OSError, ValueError) as error:
            raise WorkerLaunchError("Worker could not be started") from error

        with self._lock:
            self._process = process
        if self._cancelled.is_set():
            self.cancel()

        stdout = bytearray()
        stderr = bytearray()
        truncated = {"stdout": False, "stderr": False}

        def drain(stream, captured: bytearray, name: str) -> None:
            if stream is None:
                return
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                remaining = self.capture_limit_bytes - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(chunk) > max(remaining, 0):
                    truncated[name] = True

        stdout_thread = threading.Thread(
            target=drain,
            args=(process.stdout, stdout, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=drain,
            args=(process.stderr, stderr, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            process.wait()
        finally:
            stdout_thread.join()
            stderr_thread.join()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

        if timed_out:
            raise WorkerTimeoutError("Worker timed out")
        return ProcessResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
            stdout_truncated=truncated["stdout"],
            stderr_truncated=truncated["stderr"],
        )


async def _run_threaded_subprocess(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    capture_limit_bytes: int,
) -> ProcessResult:
    controller = _ThreadedProcess(
        command,
        cwd=cwd,
        environment=environment,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
    )
    task = asyncio.create_task(asyncio.to_thread(controller.run))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        controller.cancel()
        await asyncio.gather(asyncio.shield(task), return_exceptions=True)
        raise


def _needs_threaded_subprocess() -> bool:
    return sys.platform == "win32"


class SubprocessRunner:
    def __init__(self, capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES):
        if capture_limit_bytes < 1:
            raise ValueError("capture_limit_bytes must be positive")
        self.capture_limit_bytes = capture_limit_bytes

    async def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        extra_environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        if not command or any(not isinstance(argument, str) for argument in command):
            raise WorkerLaunchError("Invalid worker command")
        environment = os.environ.copy()
        if extra_environment:
            environment.update(extra_environment)
        if _needs_threaded_subprocess():
            return await _run_threaded_subprocess(
                command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                capture_limit_bytes=self.capture_limit_bytes,
            )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except NotImplementedError:
            return await _run_threaded_subprocess(
                command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                capture_limit_bytes=self.capture_limit_bytes,
            )
        except (OSError, ValueError) as error:
            raise WorkerLaunchError("Worker could not be started") from error

        stdout_task = asyncio.create_task(
            _read_limited(process.stdout, self.capture_limit_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_limited(process.stderr, self.capture_limit_bytes)
        )
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError as error:
                await asyncio.shield(_kill_and_wait(process))
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                raise WorkerTimeoutError("Worker timed out") from error
            stdout_result, stderr_result = await asyncio.gather(
                stdout_task,
                stderr_task,
            )
        except asyncio.CancelledError:
            await asyncio.shield(_kill_and_wait(process))
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        return ProcessResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout_result[0],
            stderr=stderr_result[0],
            stdout_truncated=stdout_result[1],
            stderr_truncated=stderr_result[1],
        )
