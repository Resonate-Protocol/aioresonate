"""Helpers for clients supporting the visualizer role."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiosendspin.models.visualizer import ClientHelloVisualizerSupport

    from .client import SendspinClient


class VisualizerClient:
    """Expose visualizer capabilities reported by the client."""

    def __init__(self, client: SendspinClient) -> None:
        """Attach to a client that exposes visualizer capabilities."""
        self.client = client
        self._logger = client._logger.getChild("visualizer")  # noqa: SLF001

    @property
    def support(self) -> ClientHelloVisualizerSupport | None:
        """Return visualizer capabilities advertised in the hello payload."""
        return self.client.info.visualizer_support
