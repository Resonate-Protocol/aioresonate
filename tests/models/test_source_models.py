"""Tests for source role protocol models."""

from __future__ import annotations

import pytest

from aiosendspin.models.core import (
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    ClientStreamEndMessage,
    ClientStreamStartMessage,
    ClientStreamStartPayload,
    ServerCommandMessage,
    ServerCommandPayload,
)
from aiosendspin.models.source import (
    ClientHelloSourceSupport,
    ClientStreamStartSource,
    SourceCommandPayload,
    SourceFeatures,
    SourceStatePayload,
)
from aiosendspin.models.types import (
    AudioCodec,
    BinaryMessageType,
    ClientMessage,
    Roles,
    ServerMessage,
    SourceCommand,
    SourceSignal,
)


def _source_hello(**overrides: object) -> ClientHelloPayload:
    kwargs: dict[str, object] = {
        "client_id": "src-1",
        "name": "Source",
        "version": 1,
        "supported_roles": [Roles.SOURCE.value],
        "source_support": ClientHelloSourceSupport(
            features=SourceFeatures(line_sense=True),
        ),
    }
    kwargs.update(overrides)
    return ClientHelloPayload(**kwargs)  # type: ignore[arg-type]


def test_source_role_and_binary_type_constants() -> None:
    """The source role and binary chunk type match the spec."""
    assert Roles.SOURCE.value == "source@v1"
    assert BinaryMessageType.SOURCE_AUDIO_CHUNK.value == 12


def test_source_support_carries_no_format_negotiation() -> None:
    """A source announces its format per stream, not in client/hello."""
    support = ClientHelloSourceSupport()
    assert support.features is None
    assert not hasattr(support, "supported_formats")
    assert "supported_formats" not in support.to_json()


def test_client_hello_source_support_round_trips_via_alias() -> None:
    """source_support serializes under the versioned alias and round-trips."""
    hello = _source_hello()
    json_str = hello.to_json()
    assert "source@v1_support" in json_str
    assert "source_support" not in json_str

    restored = ClientHelloPayload.from_json(json_str)
    assert restored.source_support is not None
    assert restored.source_support.features is not None
    assert restored.source_support.features.line_sense is True


def test_client_hello_source_support_parses_without_features() -> None:
    """A minimal source@v1_support object (no feature hints) is accepted."""
    hello = _source_hello(source_support=ClientHelloSourceSupport())
    restored = ClientHelloPayload.from_json(hello.to_json())
    assert restored.source_support is not None
    assert restored.source_support.features is None


def test_client_hello_requires_source_support_when_role_present() -> None:
    """Declaring the source role without support is rejected."""
    with pytest.raises(ValueError, match="source@v1_support"):
        ClientHelloPayload(
            client_id="src", name="s", version=1, supported_roles=[Roles.SOURCE.value]
        )


def test_client_hello_drops_source_support_when_role_absent() -> None:
    """source_support is cleared when the source role is not advertised."""
    # controller needs no support object, so this only exercises source clearing.
    hello = _source_hello(supported_roles=["controller@v1"])
    assert hello.source_support is None


def test_source_state_omits_none_signal() -> None:
    """SourceStatePayload omits the signal field when unset."""
    assert SourceStatePayload().to_dict() == {}
    assert SourceStatePayload(signal=SourceSignal.PRESENT).to_dict() == {"signal": "present"}


def test_client_state_source_round_trip() -> None:
    """client/state carries the source signal sub-object."""
    payload = ClientStatePayload(source=SourceStatePayload(signal=SourceSignal.ABSENT))
    restored = ClientStateMessage(payload=payload)
    parsed = ClientMessage.from_json(restored.to_json())
    assert isinstance(parsed, ClientStateMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.signal == SourceSignal.ABSENT


def test_server_command_source_round_trip() -> None:
    """server/command carries the source start/stop command."""
    message = ServerCommandMessage(
        payload=ServerCommandPayload(source=SourceCommandPayload(command=SourceCommand.START))
    )
    parsed = ServerMessage.from_json(message.to_json())
    assert isinstance(parsed, ServerCommandMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.command == SourceCommand.START


def test_client_stream_start_round_trip() -> None:
    """client_stream/start announces format and optional codec header."""
    message = ClientStreamStartMessage(
        payload=ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.FLAC,
                channels=2,
                sample_rate=44100,
                bit_depth=16,
                codec_header="aGVhZGVy",
            )
        )
    )
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientStreamStartMessage)
    assert parsed.payload.source.codec == AudioCodec.FLAC
    assert parsed.payload.source.codec_header == "aGVhZGVy"


def test_client_stream_end_round_trip() -> None:
    """client_stream/end round-trips with an empty payload."""
    parsed = ClientMessage.from_json(ClientStreamEndMessage().to_json())
    assert isinstance(parsed, ClientStreamEndMessage)


def test_client_stream_start_source_validates_positive_values() -> None:
    """ClientStreamStartSource validates positive dimensions."""
    with pytest.raises(ValueError, match="must be positive"):
        ClientStreamStartSource(codec=AudioCodec.PCM, channels=0, sample_rate=48000, bit_depth=16)
