"""Client spec-compliance signalling.

The server tolerates a range of non-spec-compliant client behavior for
backwards compatibility. Every such tolerance is funnelled through
``SendspinClient.flag_noncompliance`` / ``SendspinConnection._flag_noncompliance``,
which log the deviation and, when the server runs with ``strict_clients=True``,
raise ``ClientComplianceError`` to reject the client. Grep for those helpers to
enumerate the workarounds.
"""

from __future__ import annotations


class ClientComplianceError(Exception):
    """Raised when a strict-mode server rejects a non-spec-compliant client."""
