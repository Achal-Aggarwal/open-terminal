"""Bounded notebook sessions with explicit kernel ownership and cleanup."""
import asyncio
import json
import os
import time
import uuid

import aiofiles
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from jupyter_client import AsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.connect import channel_socket_types
import zmq
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from open_terminal.utils.limits import MAX_RESPONSE_BYTES, ResourceLimitError, worker_command

_IDLE_TIMEOUT = 30 * 60
_sessions = {}
_cleanup_task = None


class LimitedKernelClient(AsyncKernelClient):
    def _create_connected_socket(self, channel, identity=None):
        # Set transport budgets before connecting, so one output message cannot
        # allocate an arbitrary amount of memory before nbclient sees it.
        sock = self.context.socket(channel_socket_types[channel])
        sock.linger = 1000
        sock.setsockopt(zmq.MAXMSGSIZE, MAX_RESPONSE_BYTES)
        sock.setsockopt(zmq.RCVHWM, 8)
        if identity:
            sock.identity = identity
        sock.connect(self._make_url(channel))
        return sock


class LimitedKernelManager(AsyncKernelManager):
    def format_kernel_cmd(self, extra_arguments=None):
        return worker_command(super().format_kernel_cmd(extra_arguments))


class LimitedNotebookClient(NotebookClient):
    output_bytes = 0

    def process_message(self, msg, cell, cell_index):
        kind = msg.get("msg_type", "")
        if kind.startswith("comm"):
            return None  # Widget state is not part of the cell-output API.
        if kind in {"stream", "display_data", "update_display_data", "execute_result", "error"}:
            self.output_bytes += len(json.dumps(msg["content"], ensure_ascii=False).encode())
            if self.output_bytes > MAX_RESPONSE_BYTES:
                raise ResourceLimitError("Notebook output exceeded the byte limit; kernel stopped.")
        return super().process_message(msg, cell, cell_index)


class _Session:
    def __init__(self, session_id, path, nb, client):
        self.id, self.path, self.nb, self.client = session_id, path, nb, client
        self.busy = False
        self.created_at = self.last_used = time.time()


async def _close_client(client):
    try:
        await client._async_cleanup_kernel()
    except Exception:
        # A failed handshake can leave only the manager initialized.
        if client.km and await client.km.is_alive():
            await client.km.shutdown_kernel(now=True)


async def _destroy_session(session_id):
    session = _sessions.pop(session_id, None)
    if session:
        await _close_client(session.client)


async def _idle_cleanup_loop():
    while True:
        await asyncio.sleep(60)
        for sid, session in list(_sessions.items()):
            if not session.busy and time.time() - session.last_used > _IDLE_TIMEOUT:
                try:
                    await _destroy_session(sid)
                except Exception:
                    pass


def _ensure_cleanup_task():
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_idle_cleanup_loop())


async def shutdown_notebooks():
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        await asyncio.gather(_cleanup_task, return_exceptions=True)
        _cleanup_task = None
    await asyncio.gather(*(_destroy_session(sid) for sid in list(_sessions)), return_exceptions=True)


class CreateSessionRequest(BaseModel):
    path: str = Field(..., description="Absolute path to the .ipynb file.")


class CreateSessionResponse(BaseModel):
    id: str
    kernel: str
    status: str


class ExecuteCellRequest(BaseModel):
    cell_index: int = Field(..., description="Zero-based cell index to execute.")
    source: str | None = Field(None, max_length=MAX_RESPONSE_BYTES)


class ExecuteCellResponse(BaseModel):
    status: str
    execution_count: int | None = None
    outputs: list = Field(default_factory=list)


class SessionStatusResponse(BaseModel):
    id: str
    path: str
    kernel: str
    status: str


def create_notebooks_router(verify_api_key):
    router = APIRouter(prefix="/notebooks", tags=["notebooks"], dependencies=[Depends(verify_api_key)])

    @router.post("", response_model=CreateSessionResponse, operation_id="create_notebook_session", include_in_schema=False)
    async def create_session(req: CreateSessionRequest):
        client = None
        registered = False
        try:
            _ensure_cleanup_task()
            path = os.path.abspath(req.path)
            try:
                async with aiofiles.open(path, "rb") as source:
                    content = await source.read(MAX_RESPONSE_BYTES + 1)
            except FileNotFoundError:
                raise HTTPException(404, "Notebook not found")
            if len(content) > MAX_RESPONSE_BYTES:
                raise ResourceLimitError("Notebook exceeds the input size limit; clear saved outputs or split it.")
            try:
                nb = nbformat.reads(content.decode("utf-8"), as_version=4)
            except Exception as exc:
                raise HTTPException(400, f"Invalid notebook: {str(exc)[:500]}") from None
            if len(nb.cells) > 1000:
                raise ResourceLimitError("Notebook exceeds the cell count limit.")
            kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")
            client = LimitedNotebookClient(nb, kernel_name=kernel_name, kernel_manager_class=LimitedKernelManager,
                timeout=120, startup_timeout=30, store_widget_state=False, record_timing=False)
            try:
                async with asyncio.timeout(40):
                    client.create_kernel_manager()
                    client.km.client_class = "open_terminal.utils.notebooks.LimitedKernelClient"
                    await client.async_start_new_kernel(cwd=os.path.dirname(path))
                    await client.async_start_new_kernel_client()
            except Exception as exc:
                raise HTTPException(500, f"Failed to start kernel: {str(exc)[:500]}") from None
            sid = uuid.uuid4().hex[:12]
            _sessions[sid] = _Session(sid, path, nb, client)
            registered = True
            return CreateSessionResponse(id=sid, kernel=kernel_name, status="ready")
        finally:
            if client and not registered:
                await _close_client(client)

    @router.post("/{session_id}/execute", response_model=ExecuteCellResponse, operation_id="execute_notebook_cell", include_in_schema=False)
    async def execute_cell(session_id: str, req: ExecuteCellRequest):
        session = _sessions.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        if session.busy:
            raise HTTPException(409, "Cell already executing")
        if not 0 <= req.cell_index < len(session.nb.cells):
            raise HTTPException(400, "Cell index out of range")
        cell = session.nb.cells[req.cell_index]
        previous_source = cell.source
        if req.source is not None:
            cell.source = req.source
        if len(nbformat.writes(session.nb).encode()) > 2 * MAX_RESPONSE_BYTES:
            cell.source = previous_source
            raise ResourceLimitError("Notebook exceeds the session size limit.")
        session.busy = True
        session.last_used = time.time()
        session.client.output_bytes = sum(len(json.dumps(c.get("outputs", [])).encode()) for i, c in enumerate(session.nb.cells) if i != req.cell_index)
        try:
            async with asyncio.timeout(130):
                await session.client.async_execute_cell(cell, req.cell_index)
        except asyncio.CancelledError:
            await _destroy_session(session_id)
            raise
        except CellExecutionError:
            return ExecuteCellResponse(status="error", execution_count=cell.get("execution_count"), outputs=cell.get("outputs", []))
        except Exception as exc:
            # Errors may leave a kernel running or an IOPub queue filling.
            await _destroy_session(session_id)
            return ExecuteCellResponse(status="error", outputs=[{
                "output_type": "error", "ename": type(exc).__name__,
                "evalue": str(exc)[:1000], "traceback": [str(exc)[:1000]],
            }])
        finally:
            session.busy = False
            session.last_used = time.time()
        try:
            async with aiofiles.open(session.path, "w", encoding="utf-8") as target:
                await target.write(nbformat.writes(session.nb))
        except OSError:
            pass
        return ExecuteCellResponse(status="ok", execution_count=cell.get("execution_count"), outputs=cell.get("outputs", []))

    @router.get("/{session_id}", response_model=SessionStatusResponse, operation_id="get_notebook_session", include_in_schema=False)
    async def get_session(session_id: str):
        session = _sessions.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return SessionStatusResponse(id=session.id, path=session.path,
            kernel=session.nb.metadata.get("kernelspec", {}).get("name", "python3"), status="busy" if session.busy else "ready")

    @router.delete("/{session_id}", operation_id="delete_notebook_session", include_in_schema=False)
    async def delete_session(session_id: str):
        if session_id not in _sessions:
            raise HTTPException(404, "Session not found")
        await _destroy_session(session_id)
        return {"status": "stopped"}

    return router
