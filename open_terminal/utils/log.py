"""Bounded process logs, cursor reads, and failure-safe process finalization."""

import asyncio
from collections import deque
import json
import logging
import os
import shutil
import time
from typing import Optional

import aiofiles
import aiofiles.os

from open_terminal.env import MAX_PROCESS_LOG_SIZE, LOG_FLUSH_INTERVAL, LOG_FLUSH_BUFFER
from open_terminal.utils.limits import CHUNK_SIZE, MAX_COMMAND_SECONDS, MAX_RESPONSE_BYTES, ResourceLimitError, run_io

log = logging.getLogger(__name__)
MAX_RECORD_BYTES = CHUNK_SIZE
MAX_LOG_ENTRIES = 1000


def _records(stream):
    while line := stream.readline(MAX_RECORD_BYTES + 1):
        if len(line) > MAX_RECORD_BYTES:
            raise ResourceLimitError("Process log contains an oversized record.")
        try:
            yield json.loads(line)
        except (ValueError, UnicodeError):
            continue


def _rotate_log(path: str) -> int:
    temporary = path + ".rotating"
    try:
        with open(path, "rb") as source:
            midpoint = os.fstat(source.fileno()).st_size // 2
            offset = 0
            for record in _records(source):
                if record.get("type") == "log_rotated":
                    offset = record.get("offset", 0)
                elif record.get("type") in ("stdout", "stderr", "output"):
                    offset += 1
                if source.tell() >= midpoint:
                    break
            with open(temporary, "wb") as target:
                target.write((json.dumps({"type": "log_rotated", "offset": offset, "ts": time.time()}) + "\n").encode())
                shutil.copyfileobj(source, target, CHUNK_SIZE)
                size = target.tell()
        os.replace(temporary, path)
        return size
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class BoundedLogWriter:
    """Rotate on disk; disable logging on I/O failure while draining the child."""

    def __init__(self, file, log_path: str, *, flush_interval: float = 0, flush_buffer: int = 0):
        self._file = file
        self._log_path = log_path
        self._bytes_written = 0
        self.rotated = False
        self.error = None
        self._flush_interval = flush_interval
        self._flush_buffer = flush_buffer
        self._unflushed = 0
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()

    async def _disable(self, exc):
        self.error = str(exc)[:500]
        log.warning("Disabling process log %s: %s", self._log_path, self.error)
        await self.close()

    async def write(self, data: str) -> None:
        async with self._lock:
            if self._file is None:
                return
            try:
                size = len(data.encode("utf-8", errors="replace"))
                if size > MAX_RECORD_BYTES:
                    raise ResourceLimitError("Process log record exceeds the size limit.")
                if self._bytes_written + size > MAX_PROCESS_LOG_SIZE:
                    await self._rotate()
                await self._file.write(data)
                self._bytes_written += size
                self._unflushed += size
                if (self._flush_interval <= 0
                    or time.monotonic() - self._last_flush >= self._flush_interval
                    or self._flush_buffer > 0 and self._unflushed >= self._flush_buffer):
                    await self.flush()
            except (OSError, ValueError) as exc:
                await self._disable(exc)

    async def flush(self) -> None:
        if self._file is not None:
            try:
                await self._file.flush()
                self._unflushed = 0
                self._last_flush = time.monotonic()
            except (OSError, ValueError) as exc:
                await self._disable(exc)

    async def close(self):
        file, self._file = self._file, None
        if file is not None:
            try:
                await file.close()
            except (OSError, ValueError):
                pass

    async def _rotate(self):
        await self._file.flush()
        await self._file.close()
        self._file = None
        self._bytes_written = await run_io(_rotate_log, self._log_path)
        self._file = await aiofiles.open(self._log_path, "a", encoding="utf-8")
        self.rotated = True


async def log_process(background_process) -> None:
    writer = None
    runner = background_process.runner
    try:
        if background_process.log_path:
            try:
                await aiofiles.os.makedirs(os.path.dirname(background_process.log_path), exist_ok=True)
                file = await aiofiles.open(background_process.log_path, "w", encoding="utf-8")
                writer = BoundedLogWriter(file, background_process.log_path,
                    flush_interval=LOG_FLUSH_INTERVAL, flush_buffer=LOG_FLUSH_BUFFER)
                await writer.write(json.dumps({"type": "start", "command": background_process.command[:4096], "pid": runner.pid, "ts": time.time()}) + "\n")
            except OSError as exc:
                log.warning("Cannot open process log: %s", exc)
        async with asyncio.timeout(MAX_COMMAND_SECONDS):
            await runner.read_output(writer)
            background_process.exit_code = await runner.wait()
    except asyncio.CancelledError:
        background_process.status = "killed"
        runner.kill(force=True)
        raise
    except Exception as exc:
        background_process.status = "error"
        background_process.error = ("Command exceeded its execution deadline." if isinstance(exc, TimeoutError) else str(exc)[:500])
        runner.kill(force=True)
        log.warning("Command %s failed: %s", background_process.id, background_process.error)
    finally:
        try:
            background_process.exit_code = await runner.wait()
        finally:
            runner.close()
            if background_process.status == "running":
                background_process.status = "done"
            background_process.finished_at = time.time()
            if writer:
                try:
                    await writer.write(json.dumps({"type": "end", "exit_code": background_process.exit_code, "log_rotated": writer.rotated, "ts": time.time()}) + "\n")
                    await writer.flush()
                finally:
                    await writer.close()
                if writer.error:
                    background_process.error = "Output logging stopped: " + writer.error


def _read_log(path, offset, tail):
    selected = deque()
    total = 0
    size = 0
    truncated = False
    next_offset = offset
    limit = min(tail, MAX_LOG_ENTRIES) if tail is not None else MAX_LOG_ENTRIES
    try:
        source = open(path, "rb")
    except FileNotFoundError:
        return [], 0, False
    with source:
        for record in _records(source):
            kind = record.get("type")
            if kind == "log_rotated":
                total = record.get("offset", 0)
                truncated |= offset < total
                next_offset = max(next_offset, total)
            elif kind in ("stdout", "stderr", "output"):
                index = total
                total += 1
                if index < offset:
                    continue
                entry = {"type": kind, "data": record["data"]}
                entry_size = len(json.dumps(entry, ensure_ascii=False).encode())
                if entry_size > MAX_RESPONSE_BYTES:
                    raise ResourceLimitError("A log entry exceeds the response byte limit.")
                if tail is None and (len(selected) >= limit or size + entry_size > MAX_RESPONSE_BYTES):
                    return list(selected), index, True
                selected.append(entry)
                size += entry_size
                while len(selected) > limit or size > MAX_RESPONSE_BYTES:
                    removed = selected.popleft()
                    size -= len(json.dumps(removed, ensure_ascii=False).encode())
                    truncated = True
                next_offset = total
    return list(selected), max(next_offset, total), truncated


async def read_log(log_path: Optional[str], offset: int = 0, tail: Optional[int] = None) -> tuple[list[dict], int, bool]:
    """Return a byte-bounded page and the next unread absolute cursor.

    Explicit tail reads select the newest bounded entries and may skip older
    output. Default reads paginate forward without silently losing entries.
    """
    if not log_path:
        return [], 0, False
    return await run_io(_read_log, log_path, offset, tail)
