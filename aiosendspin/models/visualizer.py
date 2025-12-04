"""
Visualizer messages for the Sendspin protocol.

This module contains messages specific to clients with the visualizer role, which
create visual representations of the audio being played. Visualizer clients receive
audio analysis data like FFT information that corresponds to the current audio timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin


# Client -> Server: client/hello visualizer support object
@dataclass
class ClientHelloVisualizerSupport(DataClassORJSONMixin):
    """Visualizer support configuration - only if visualizer role is set."""

    buffer_capacity: int
    """
    Max size in bytes of visualization data messages in the buffer that are yet to be displayed.
    """

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {self.buffer_capacity}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: stream/start visualizer object
@dataclass
class StreamStartVisualizer(DataClassORJSONMixin):
    """Visualizer object in stream/start message."""
