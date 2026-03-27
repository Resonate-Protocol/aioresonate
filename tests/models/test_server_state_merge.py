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


def test_server_state_merge_null_clears_existing_field() -> None:
    """Per the spec, fields set to null should be cleared from state."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=100,
                title="Song Title",
                artist="Artist Name",
                repeat=RepeatMode.ALL,
            )
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=200,
                title=None,
                artist=None,
            )
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is not None
    assert merged.payload.metadata.timestamp == 200
    # Explicitly set to None → should be cleared
    assert merged.payload.metadata.title is None
    assert merged.payload.metadata.artist is None
    # Not included in delta (UndefinedField) → should be preserved
    assert merged.payload.metadata.repeat == RepeatMode.ALL


def test_server_state_merge_null_clears_nested_progress() -> None:
    """Setting progress to None should clear it, not preserve the old value."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=100,
                progress=Progress(
                    track_progress=30_000,
                    track_duration=213_000,
                    playback_speed=1_000,
                ),
            )
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=200,
                progress=None,
            )
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is not None
    assert merged.payload.metadata.timestamp == 200
    assert merged.payload.metadata.progress is None
