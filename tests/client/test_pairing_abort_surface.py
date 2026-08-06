"""Non-closing pairing aborts are surfaced to registered SDK listeners."""

from __future__ import annotations

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import PairAbortReason, Roles
from aiosendspin.noise.pairing import RemotePairingAbortError
from tests.conftest import make_sdk_client


async def test_non_closing_pairing_abort_notifies_listener() -> None:
    """A non-closing abort during pairing reaches the pairing-abort listener with its reason."""
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    reasons: list[PairAbortReason] = []
    sdk.add_pairing_abort_listener(reasons.append)
    connection = SendspinConnection(sdk)

    async def _abort() -> str | None:
        raise RemotePairingAbortError(PairAbortReason.LOCKED_OUT)

    connection._run_pairing_protocol = _abort  # type: ignore[method-assign]  # noqa: SLF001
    await connection._pair()  # noqa: SLF001

    assert reasons == [PairAbortReason.LOCKED_OUT]
