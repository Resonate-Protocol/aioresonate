"""Tests for ``DeviceInfo.connections`` round-trip across the Sendspin protocol wire format."""

from __future__ import annotations

import pytest

from aiosendspin.models.core import (
    ClientHelloPayload,
    DeviceInfo,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec


def _player_support() -> ClientHelloPlayerSupport:
    """Build a minimal valid ``player@v1_support`` block for hello payloads."""
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2
            )
        ],
        buffer_capacity=100_000,
        supported_commands=[],
    )


def test_device_info_connections_default_none() -> None:
    """A bare ``DeviceInfo`` defaults ``connections`` to ``None`` so the
    field is omitted from the on-wire JSON for clients that don't set it."""
    info = DeviceInfo(product_name="Test", manufacturer="Acme")
    assert info.connections is None


def test_device_info_connections_omitted_from_json_when_none() -> None:
    """``omit_none = True`` keeps the wire format unchanged for old clients."""
    info = DeviceInfo(product_name="Test", manufacturer="Acme")
    payload = info.to_dict()
    assert "connections" not in payload


def test_device_info_connections_round_trip_bluetooth_mac() -> None:
    """A BT bridge can advertise the speaker's MAC and parse it back."""
    info = DeviceInfo(
        product_name="ENEBY20",
        manufacturer="Sendspin BT Bridge",
        connections=[("bluetooth", "AA:BB:CC:DD:EE:FF")],
    )
    restored = DeviceInfo.from_dict(info.to_dict())
    assert restored.connections == [("bluetooth", "AA:BB:CC:DD:EE:FF")]


def test_device_info_connections_round_trip_arbitrary_type() -> None:
    """Connection types are pure strings — protocol does NOT filter to a known
    taxonomy. Future transports (Zigbee EUI-64, Matter device-id, custom)
    pass through verbatim."""
    info = DeviceInfo(
        product_name="Future Bridge",
        connections=[
            ("zigbee", "00:0d:6f:00:00:00:11:22"),
            ("matter", "vendor-1234/product-5678"),
            ("custom_proto", "any-string"),
        ],
    )
    restored = DeviceInfo.from_dict(info.to_dict())
    assert restored.connections == [
        ("zigbee", "00:0d:6f:00:00:00:11:22"),
        ("matter", "vendor-1234/product-5678"),
        ("custom_proto", "any-string"),
    ]


def test_device_info_connections_multiple_per_device() -> None:
    """A device with multiple hardware identities advertises all of them."""
    info = DeviceInfo(
        connections=[
            ("mac", "AA:BB:CC:DD:EE:FF"),
            ("bluetooth", "11:22:33:44:55:66"),
        ],
    )
    restored = DeviceInfo.from_dict(info.to_dict())
    assert restored.connections == [
        ("mac", "AA:BB:CC:DD:EE:FF"),
        ("bluetooth", "11:22:33:44:55:66"),
    ]


def test_device_info_connections_empty_list_round_trips() -> None:
    """Explicit empty list survives — distinct from absent (None)."""
    info = DeviceInfo(connections=[])
    restored = DeviceInfo.from_dict(info.to_dict())
    assert restored.connections == []


def test_device_info_old_client_payload_parses() -> None:
    """A payload from an older client without the ``connections`` key still
    deserializes — backwards compat for protocol consumers."""
    legacy_payload = {
        "product_name": "Old client",
        "manufacturer": "Vendor",
        "software_version": "1.0.0",
    }
    info = DeviceInfo.from_dict(legacy_payload)
    assert info.product_name == "Old client"
    assert info.connections is None


def test_client_hello_carries_device_info_connections() -> None:
    """End-to-end: ``client/hello`` with ``device_info.connections`` round-trips."""
    payload = ClientHelloPayload(
        client_id="bridge-001",
        name="Sendspin BT Bridge",
        version=1,
        supported_roles=["player@v1"],
        device_info=DeviceInfo(
            product_name="Sendspin BT Bridge v9.9.9",
            manufacturer="HostName",
            connections=[("bluetooth", "AA:BB:CC:DD:EE:FF")],
        ),
        player_support=_player_support(),
    )
    restored = ClientHelloPayload.from_dict(payload.to_dict())
    assert restored.device_info is not None
    assert restored.device_info.connections == [
        ("bluetooth", "AA:BB:CC:DD:EE:FF")
    ]


@pytest.mark.parametrize(
    "raw_value",
    [
        "AA:BB:CC:DD:EE:FF",
        "aa-bb-cc-dd-ee-ff",
        "AABBCCDDEEFF",
        "00:0d:6f:00:00:00:11:22",  # 8-byte EUI-64 — wider than MAC
    ],
)
def test_device_info_connections_value_passthrough(raw_value: str) -> None:
    """Whatever raw value the bridge sends is preserved verbatim — protocol
    layer does not normalize. Normalization is the receiving server's
    responsibility (e.g. MA's ``DeviceInfo.add_connection`` helper does
    canonical lowercase-with-colons for MAC-shaped values)."""
    info = DeviceInfo(connections=[("bluetooth", raw_value)])
    restored = DeviceInfo.from_dict(info.to_dict())
    assert restored.connections == [("bluetooth", raw_value)]
