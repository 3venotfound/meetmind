import asyncio
import os
import sys
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
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
