"""Regression tests for server/state message merging."""

from __future__ import annotations

from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import MediaCommand, RepeatMode, UndefinedField


def test_server_state_absent_role_omitted_from_wire() -> None:
    """A role left unset is UndefinedField and omitted from serialization."""
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=100, title="X"))
    encoded = payload.to_dict()
    assert "color" not in encoded
    assert "controller" not in encoded
    assert isinstance(ServerStatePayload.from_dict(encoded).color, UndefinedField)


def test_server_state_whole_role_null_round_trips() -> None:
    """A whole-role object set to null serializes as null and decodes to None."""
    payload = ServerStatePayload(metadata=None)
    assert payload.to_dict() == {"metadata": None}
    decoded = ServerStatePayload.from_dict({"metadata": None})
    assert decoded.metadata is None
    assert isinstance(decoded.color, UndefinedField)
    assert isinstance(decoded.controller, UndefinedField)


def test_server_state_merge_whole_role_null_clears_role() -> None:
    """A whole-role null clears that role; an absent role keeps its existing state."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
        )
    )
    incoming = ServerStateMessage(payload=ServerStatePayload(metadata=None))

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is None


def test_server_state_merge_absent_role_preserved() -> None:
    """An incoming delta that omits a role leaves the existing role state intact."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            color=None,
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is not None
    assert merged.payload.metadata.title == "Song Title"
    assert merged.payload.color is None


def test_server_state_merge_preserves_metadata_fields_omitted_by_undefined() -> None:
    """Keep existing metadata fields when a later delta omits them with UndefinedField."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=100,
                title="Song Title",
                album="Some Album",
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
    assert merged.payload.metadata.title == "Song Title"
    assert merged.payload.metadata.album == "Some Album"
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
                album="Some Album",
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
    assert merged.payload.metadata.album == "Some Album"


def test_server_state_merge_controller_overwrites_repeat_and_shuffle() -> None:
    """Incoming controller state overwrites existing repeat/shuffle (required fields)."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            controller=ControllerStatePayload(
                supported_commands=[MediaCommand.PLAY],
                volume=50,
                muted=False,
                repeat=RepeatMode.OFF,
                shuffle=False,
            )
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            controller=ControllerStatePayload(
                supported_commands=[MediaCommand.PLAY],
                volume=50,
                muted=False,
                repeat=RepeatMode.ALL,
                shuffle=True,
            )
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.controller is not None
    assert merged.payload.controller.repeat == RepeatMode.ALL
    assert merged.payload.controller.shuffle is True


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
