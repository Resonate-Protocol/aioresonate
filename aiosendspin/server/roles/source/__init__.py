"""Source role - client and group level."""

from aiosendspin.models.source import ClientHelloSourceSupport
from aiosendspin.server.roles.registry import (
    RoleSupportSpec,
    register_group_role,
    register_role,
    register_role_support_spec,
)
from aiosendspin.server.roles.source.decoder import SourceDecoder
from aiosendspin.server.roles.source.events import (
    SourceEvent,
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from aiosendspin.server.roles.source.group import SourceGroupRole
from aiosendspin.server.roles.source.stream import SourceAudioChunk, SourceAudioStream
from aiosendspin.server.roles.source.v1 import SourceRoleState, SourceV1Role

register_group_role("source", SourceGroupRole)
register_role("source@v1", lambda client: SourceV1Role(client=client))
register_role_support_spec(
    "source",
    RoleSupportSpec(
        parse_support=ClientHelloSourceSupport.from_dict,
    ),
)

__all__ = [
    "SourceAudioChunk",
    "SourceAudioStream",
    "SourceDecoder",
    "SourceEvent",
    "SourceGroupRole",
    "SourceRoleState",
    "SourceSignalChangedEvent",
    "SourceStreamEndedEvent",
    "SourceStreamStartedEvent",
    "SourceV1Role",
]
