"""Utility functions for aiosendspin."""

from __future__ import annotations

import socket


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
