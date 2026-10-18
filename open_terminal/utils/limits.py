"""Memory budgets shared by HTTP handlers and subprocess workers."""

import asyncio
import os
from pathlib import Path
import sys

from fastapi import HTTPException
from starlette.responses import JSONResponse

from open_terminal import config


def positive_setting(name: str, default: int) -> int:
    value = int(os.environ.get(f"OPEN_TERMINAL_{name.upper()}", config.get(name, default)))
    if value <= 0:
        raise ValueError(f"OPEN_TERMINAL_{name.upper()} must be positive")
    return value


CHUNK_SIZE = 64 * 1024
MAX_RESPONSE_BYTES = positive_setting("max_response_bytes", 1024 * 1024)
MAX_REQUEST_BYTES = positive_setting("max_request_bytes", 2 * 1024 * 1024)
MAX_TRANSFER_BYTES = positive_setting("max_transfer_bytes", 1024 * 1024 * 1024)
MAX_COMMAND_SECONDS = positive_setting("max_command_seconds", 3600)


class ResourceLimitError(ValueError):
    pass


async def run_io(function, *args):
    """Keep admission/cleanup ownership until a cancelled thread actually stops."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise


class ResourceLimitsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            from urllib.parse import parse_qsl, urlencode
            params = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
            cleaned = [(key, value) for key, value in params if value.lower() != "null"]
            if len(cleaned) != len(params):
                scope = {**scope, "query_string": urlencode(cleaned).encode()}
        if scope["type"] != "http" or (scope["path"] == "/health" or scope["path"].startswith(("/files/", "/skills"))):
            return await self.app(scope, receive, send)
        limit = MAX_TRANSFER_BYTES if scope["path"] == "/files/upload" or scope["path"].startswith("/proxy/") else MAX_REQUEST_BYTES
        headers = dict(scope.get("headers", []))
        try:
            size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            size = limit + 1
        if size < 0 or size > limit:
            return await JSONResponse({"detail": "Request body exceeds the size limit."}, status_code=413)(scope, receive, send)
        received = 0
        quoted = escaped = False
        depth = tokens = 0
        is_json = limit == MAX_REQUEST_BYTES

        async def limited_receive():
            nonlocal received, quoted, escaped, depth, tokens
            message = await receive()
            body = message.get("body", b"")
            received += len(body)
            if received > limit:
                raise HTTPException(413, "Request body exceeds the size limit.")
            if is_json:
                # Check structure before json.loads can expand a small body into
                # hundreds of thousands of dicts/lists. String contents are free.
                for byte in body:
                    if quoted:
                        if escaped:
                            escaped = False
                        elif byte == 92:
                            escaped = True
                        elif byte == 34:
                            quoted = False
                    elif byte == 34:
                        quoted = True
                    elif byte in (123, 91):
                        depth += 1
                        tokens += 1
                    elif byte in (125, 93):
                        depth -= 1
                    elif byte in (44, 58):
                        tokens += 1
                    if depth > 64 or tokens > 10_000:
                        raise HTTPException(413, "JSON structure exceeds the complexity limit.")
            return message

        await self.app(scope, limited_receive, send)


def worker_command(command: list[str], *, memory_limit: int = 0) -> list[str]:
    """Apply workload isolation before exec, without unsafe preexec_fn.

    Address-space limits suit our document workers. General runtimes (Node,
    JVM, scientific Python) reserve large virtual ranges and need a cgroup
    budget on actual memory instead.
    """
    if os.name == "nt":
        return command
    cgroup = os.environ.get("OPEN_TERMINAL_WORKLOAD_CGROUP", "")
    if cgroup:
        path = Path(cgroup)
        if (path / "memory.max").read_text().strip() == "max":
            raise ValueError("OPEN_TERMINAL_WORKLOAD_CGROUP must have a finite memory.max")
    if not memory_limit and not cgroup:
        return command
    return [sys.executable, str(Path(__file__).with_name("worker.py")), str(memory_limit), cgroup, *command]


async def run_worker(command, payload=None, *, request=None, timeout=60, **options):
    """Run document work with cancellation and process-group cleanup."""
    import signal
    import subprocess

    worker_env = dict(options.pop("env", os.environ))
    # Document extraction does not need BLAS parallelism.
    worker_env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="2")
    process = await asyncio.create_subprocess_exec(
        *worker_command(command), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt", env=worker_env, **options,
    )

    async def exchange():
        if payload:
            process.stdin.write(payload)
            await process.stdin.drain()
        process.stdin.close()
        output = bytearray()
        while chunk := await process.stdout.read(CHUNK_SIZE):
            output.extend(chunk)
        await process.wait()
        if process.returncode:
            raise HTTPException(422, "Document worker failed. Check that the document libraries are installed and the files are readable.")
        return bytes(output)

    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    task = asyncio.create_task(exchange())
    disconnect = asyncio.create_task(disconnected()) if request else None
    try:
        done, _ = await asyncio.wait(
            {task, disconnect} if disconnect else {task},
            timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise HTTPException(408, "Document operation exceeded its execution deadline.")
        if disconnect and disconnect in done:
            raise HTTPException(499, "Document operation cancelled.")
        return await task
    finally:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.returncode is None:
            process.kill()
        task.cancel()
        if disconnect:
            disconnect.cancel()
        await asyncio.gather(*([task, disconnect] if disconnect else [task]), return_exceptions=True)
        process.stdin.close()
        # A paused stdout transport can keep Process.wait() pending even
        # after SIGKILL. Discard remaining bounded pipe chunks before reaping.
        while await process.stdout.read(CHUNK_SIZE):
            pass
        await process.wait()
