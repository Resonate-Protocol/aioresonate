"""Regression tests for server/state message merging."""

from __future__ import annotations

from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import RepeatMode


def test_server_state_merge_preserves_metadata_fields_omitted_by_undefined() -> None:
    """Keep repeat/shuffle when a later metadata delta omits them with UndefinedField."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=100,
                repeat=RepeatMode.ALL,
                shuffle=True,
            )
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=200,
                progress=Progress(
                    track_progress=1_234,
                    track_duration=5_678,
                    playback_speed=1_000,
                ),
            )
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is not None
    assert merged.payload.metadata.timestamp == 200
    assert merged.payload.metadata.repeat == RepeatMode.ALL
    assert merged.payload.metadata.shuffle is True
    assert merged.payload.metadata.progress == Progress(
        track_progress=1_234,
        track_duration=5_678,
        playback_speed=1_000,
    )
