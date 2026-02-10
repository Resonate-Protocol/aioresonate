"""Tests for ClientListener defaults."""

from aiohttp import web

from aiosendspin.client.listener import DEFAULT_PORT, ClientListener


async def _noop_handler(_ws: web.WebSocketResponse) -> None:
    """No-op websocket handler for listener construction tests."""


def test_client_listener_default_port_matches_spec() -> None:
    """ClientListener should default to the spec-recommended listener port."""
    listener = ClientListener(client_id="client-1", on_connection=_noop_handler)
    assert DEFAULT_PORT == 8928
    assert listener.port == 8928
