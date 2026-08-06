"""The client advertises dynamic-PIN out-channels it can actually use."""

from __future__ import annotations

from unittest.mock import AsyncMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import PairMethod, Roles
from tests.conftest import make_sdk_client


async def test_dynamic_pin_out_channels_reflect_display_capability() -> None:
    """out_channels lists ``display`` only when a pin_display handler is configured."""
    with_display = make_sdk_client(
        client_name="c", roles=[Roles.CONTROLLER], pin_display=AsyncMock()
    )
    without_display = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])

    with_desc = await SendspinConnection(with_display)._pair_method_descriptor(  # noqa: SLF001
        PairMethod.DYNAMIC_PIN
    )
    without_desc = await SendspinConnection(without_display)._pair_method_descriptor(  # noqa: SLF001
        PairMethod.DYNAMIC_PIN
    )

    assert with_desc.out_channels == ["display"]
    assert without_desc.out_channels == []
