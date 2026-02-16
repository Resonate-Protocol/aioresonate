"""Tests for source@v1 protocol models."""

from __future__ import annotations

from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientCommandPayload,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    InputStreamEndMessage,
    InputStreamEndPayload,
    InputStreamRequestFormatMessage,
    InputStreamRequestFormatPayload,
    InputStreamStartMessage,
    InputStreamStartPayload,
    ServerCommandMessage,
    ServerCommandPayload,
)
from aiosendspin.models.source import (
    ClientHelloSourceSupport,
    InputStreamRequestFormatSource,
    InputStreamStartSource,
    SourceClientCommandPayload,
    SourceCommandPayload,
    SourceFeatures,
    SourceFormat,
    SourceStatePayload,
)
from aiosendspin.models.types import (
    AudioCodec,
    ClientMessage,
    Roles,
    ServerMessage,
    SourceClientCommand,
    SourceCommand,
    SourceSignalType,
    SourceStateType,
)


def test_source_hello_roundtrip() -> None:
    """Round-trip client/hello with source support payload."""
    payload = ClientHelloPayload(
        client_id="source-1",
        name="Source One",
        version=1,
        supported_roles=[Roles.SOURCE.value],
        source_support=ClientHelloSourceSupport(
            supported_formats=[
                SourceFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            features=SourceFeatures(level=True, line_sense=True),
        ),
    )
    message = ClientHelloMessage(payload=payload)
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientHelloMessage)
    assert parsed.payload.source_support is not None


def test_source_state_roundtrip() -> None:
    """Round-trip client/state with source state payload."""
    payload = ClientStatePayload(
        source=SourceStatePayload(
            state=SourceStateType.STREAMING,
            level=0.5,
            signal=SourceSignalType.PRESENT,
        )
    )
    message = ClientStateMessage(payload=payload)
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientStateMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.state == SourceStateType.STREAMING


def test_source_command_roundtrip() -> None:
    """Round-trip server/command source start payload."""
    payload = ServerCommandPayload(source=SourceCommandPayload(command=SourceCommand.START))
    message = ServerCommandMessage(payload=payload)
    parsed = ServerMessage.from_json(message.to_json())
    assert isinstance(parsed, ServerCommandMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.command == SourceCommand.START


def test_source_client_command_roundtrip() -> None:
    """Round-trip client/command source lifecycle event payload."""
    payload = ClientCommandPayload(
        source=SourceClientCommandPayload(command=SourceClientCommand.STARTED)
    )
    message = ClientCommandMessage(payload=payload)
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientCommandMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.command == SourceClientCommand.STARTED


def test_input_stream_start_roundtrip() -> None:
    """Round-trip input_stream/start source format payload."""
    message = InputStreamStartMessage(
        payload=InputStreamStartPayload(
            source=InputStreamStartSource(
                codec=AudioCodec.OPUS,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
                codec_header="AQID",
            )
        )
    )
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, InputStreamStartMessage)
    assert parsed.payload.source.codec == AudioCodec.OPUS


def test_input_stream_request_format_roundtrip() -> None:
    """Round-trip input_stream/request-format source payload."""
    message = InputStreamRequestFormatMessage(
        payload=InputStreamRequestFormatPayload(
            source=InputStreamRequestFormatSource(sample_rate=44100)
        )
    )
    parsed = ServerMessage.from_json(message.to_json())
    assert isinstance(parsed, InputStreamRequestFormatMessage)
    assert parsed.payload.source.sample_rate == 44100


def test_input_stream_end_roundtrip() -> None:
    """Round-trip input_stream/end payload."""
    message = InputStreamEndMessage(payload=InputStreamEndPayload())
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, InputStreamEndMessage)
