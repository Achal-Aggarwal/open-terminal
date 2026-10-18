"""Bounded directory traversal and text snippets for file searches."""
import os
import subprocess
import threading

from open_terminal.utils.limits import CHUNK_SIZE, MAX_SEARCH_ENTRIES, ResourceLimitError


def walk_entries(fs, root, show_hidden=True):
    """Yield at most the scan budget; never materialize an unlimited directory."""
    count = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if not show_hidden and entry.name.startswith(".") or not fs.is_path_allowed(entry.path):
                    continue
                count += 1
                if count > MAX_SEARCH_ENTRIES:
                    raise ResourceLimitError("Directory tree exceeds the search limit; narrow the path.")
                is_dir = entry.is_dir(follow_symlinks=False)
                yield entry.path, "directory" if is_dir else "file"
                if is_dir:
                    stack.append(entry.path)


def search_entries(fs, root, show_hidden=False):
    """Honor git ignores when available, with bounded subprocess output."""
    try:
        process = subprocess.Popen(
            ["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z", "--", "."],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError:
        yield from walk_entries(fs, root, show_hidden)
        return
    deadline = threading.Timer(10, process.kill)
    deadline.start()
    try:
        data = bytearray()
        while chunk := process.stdout.read(CHUNK_SIZE):
            if len(data) + len(chunk) > 2 * 1024 * 1024:
                raise ResourceLimitError("Git file listing exceeds the search limit; narrow the path.")
            data.extend(chunk)
        process.wait(timeout=10)
    finally:
        deadline.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
    if process.returncode:
        yield from walk_entries(fs, root, show_hidden)
        return
    seen = set()
    for raw in data.split(b"\0"):
        if not raw:
            continue
        rel = os.fsdecode(bytes(raw))
        if not show_hidden and any(part.startswith(".") for part in rel.replace("\\", "/").split("/")):
            continue
        path = os.path.normpath(os.path.join(root, rel))
        candidates = [path]
        parent = os.path.dirname(path)
        while parent != root and parent.startswith(root + os.sep):
            candidates.append(parent)
            parent = os.path.dirname(parent)
        for candidate in candidates:
            if candidate in seen or not fs.is_path_allowed(candidate):
                continue
            seen.add(candidate)
            if len(seen) > MAX_SEARCH_ENTRIES:
                raise ResourceLimitError("Directory tree exceeds the search limit; narrow the path.")
            yield candidate, "directory" if os.path.isdir(candidate) else "file"


def text_lines(path):
    with open(path, encoding="utf-8", errors="strict") as source:
        number = 0
        while line := source.readline(CHUNK_SIZE + 1):
            number += 1
            if len(line) > CHUNK_SIZE:
                # Do not retain or search minified/oversized lines.
                while line and not line.endswith("\n"):
                    line = source.readline(CHUNK_SIZE + 1)
                continue
            yield number, line


def snippet(text, index):
    start = max(0, index - 120)
    return text[start:start + 512].rstrip("\r\n")
