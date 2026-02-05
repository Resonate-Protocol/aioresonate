"""Utility functions for aiosendspin."""

from __future__ import annotations

import asyncio
import socket
import sys
from collections.abc import Coroutine
from typing import Any

# Check if eager_start is supported (Python 3.12+)
_SUPPORTS_EAGER_START = sys.version_info >= (3, 12)

TASKS: set[asyncio.Task[Any]] = set()


def create_task[T](
    coro: Coroutine[None, None, T],
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    name: str | None = None,
    eager_start: bool = True,
) -> asyncio.Task[T]:
    """Create an asyncio task with eager_start=True by default.

    This wrapper ensures tasks begin executing immediately rather than
    waiting for the next event loop iteration, improving performance
    and reducing latency (when supported by the Python version).

    Note: eager_start is only supported in Python 3.12+. On older versions,
    this parameter is ignored and tasks behave normally.

    Args:
        coro: The coroutine to run as a task.
        loop: Optional event loop to use. If None, uses the running loop.
        name: Optional name for the task (for debugging).
        eager_start: Whether to start the task eagerly (default: True).
                     Only used if Python version supports it.

    Returns:
        The created asyncio Task.
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    if _SUPPORTS_EAGER_START and eager_start:
        # Use Task constructor directly - it supports eager_start and schedules automatically
        task = asyncio.Task(coro, loop=loop, name=name, eager_start=True)
    else:
        task = loop.create_task(coro, name=name)

    if task.done():
        return task

    TASKS.add(task)
    task.add_done_callback(TASKS.discard)
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    return task


def get_local_ip() -> str | None:
    """Get a local IP address that can be used for mDNS advertising.

    Returns the IP address of the interface that would be used to connect
    to an external address, or None if no network is available.
    """
    try:
        # Create a UDP socket and connect to an external address
        # This doesn't send any data, just determines which interface would be used
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            result: str = s.getsockname()[0]
            return result
    except OSError:
        return None
