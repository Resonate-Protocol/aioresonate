"""Constrained pairing-store fakes shared across the noise/client/integration suites."""

from __future__ import annotations

from aiosendspin.noise.trust_store import InMemoryClientPairingStore, StorageReport


class ExhaustedClientStore(InMemoryClientPairingStore):
    """Client store that cannot persist new records (exercises the shared-PSK fallback)."""

    async def can_store_record(self) -> bool:
        """Refuse: there is no capacity for a new record."""
        return False


class BoundedClientStore(InMemoryClientPairingStore):
    """Client store with a fixed four-slot record budget (one slot per record)."""

    async def storage_accounting(self) -> StorageReport:
        """Report a four-slot budget, one slot consumed per stored record."""
        used = len(await self.list_records())
        return StorageReport(capacity=4, free=4 - used, cost_individual=1, cost_shared=1)
