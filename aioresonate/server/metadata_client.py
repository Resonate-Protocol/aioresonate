"""Helpers for clients supporting the metadata role."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aioresonate.models.metadata import ClientHelloMetadataSupport

    from .client import ResonateClient


class MetadataClient:
    """Expose metadata capabilities reported by the client."""

    def __init__(self, client: ResonateClient) -> None:
        """Attach to a client that exposes metadata capabilities."""
        self.client = client
        self._logger = client._logger.getChild("metadata")  # noqa: SLF001

    @property
    def support(self) -> ClientHelloMetadataSupport | None:
        """Return metadata capabilities advertised in the hello payload."""
        return self.client.info.metadata_support
