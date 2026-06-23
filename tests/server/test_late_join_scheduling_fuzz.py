"""Seeded yield-injection stress tests for late-join scheduling races.

asyncio is single-threaded, so a slow CPU cannot create data races; it changes
the order in which coroutines resume at ``await`` points. These tests perturb
that ordering by injecting randomized ``asyncio.sleep`` jitter into the hot
late-join paths (``commit_audio``, the catch-up task, encode, delivery), then
assert the same cross-member sync invariant as
``test_late_joiner_shares_group_timeline``: a late joiner must land on the
existing member's exact timeline, with no anchor offset.

The join delay (``get_join_delay_s``) is left at the _DummyRole default of 0.0
so this isolates SCHEDULING races from the separate, already-known bug that the
1s join-stabilization delay is never applied in production.
"""

from __future__ import annotations

import asyncio
import random
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest

from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import AudioRequirements
from tests.server.test_push_stream_behavior import _DummyClient, _DummyGroup, _DummyRole

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

SEEDS = [1, 2, 3, 5, 8, 13, 21, 34]

# Optional ad-hoc deep sweep. Keep at 0 in normal development/CI; set to e.g.
# 500 locally to crank the interleaving search.
LARGE_SWEEP_SEED_COUNT = 0
LARGE_SWEEP_SEEDS = list(range(100, 100 + LARGE_SWEEP_SEED_COUNT))

# Paths wrapped with jitter. Each is awaited from another awaited path, so
# inserting sleeps here reorders the commit / catch-up / delivery interleaving.
_JITTER_TARGETS = (
    "commit_audio",
    "_start_catchup_encoding",
    "_encode_catchup_sequence",
    "_deliver_audio_to_roles",
)


class _TransformerA:
    pending_timestamp_us: int | None = None

    @property
    def frame_duration_us(self) -> int:
        return 25_000

    def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
        return [(pcm, 25_000)]

    def flush(self) -> list[tuple[bytes, int]]:
        return []

    def get_header(self) -> bytes | None:
        return None

    def reset(self) -> None:
        return


class _TransformerB(_TransformerA):
    """Distinct type so the joiner gets its own TransformKey and a catch-up task."""


def _player_requirements(transformer: Any) -> AudioRequirements:
    return AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=transformer,
        channel_id=MAIN_CHANNEL,
        frame_duration_us=25_000,
    )


def _install_jitter(stream: PushStream, rng: random.Random, jitter_s: float) -> None:
    """Wrap hot async paths so each yields a random amount at its boundaries."""

    def make(orig: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if jitter_s:
                await asyncio.sleep(rng.uniform(0, jitter_s))
            result = await orig(*args, **kwargs)
            if jitter_s:
                await asyncio.sleep(rng.uniform(0, jitter_s))
            return result

        return wrapped

    for name in _JITTER_TARGETS:
        setattr(stream, name, make(getattr(stream, name)))


async def _commit(stream: PushStream) -> None:
    stream.prepare_audio(
        bytes(7200),  # 25ms @ 48kHz stereo 24-bit
        AudioFormat(sample_rate=48000, bit_depth=24, channels=2),
    )
    await stream.commit_audio()


async def _run_scenario(seed: int) -> None:
    """Run one seeded late-join scenario under injected scheduling jitter."""
    rng = random.Random(seed)  # noqa: S311
    jitter_s = rng.choice([0.0, 0.0001, 0.0005, 0.001])
    pre_join_commits = rng.randint(1, 4)
    post_join_commits = rng.randint(1, 4)
    # How many times to pump the loop right after the join, before firing the
    # post-join commits: varies whether catch-up partially completes first.
    post_join_pumps = rng.randint(0, 3)

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(_player_requirements(_TransformerA()))
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    _install_jitter(stream, rng, jitter_s)

    for _ in range(pre_join_commits):
        await _commit(stream)

    role2 = _DummyRole(_player_requirements(_TransformerB()), replay_from_pcm_cache=True)
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    for _ in range(post_join_pumps):
        await asyncio.sleep(0)

    for _ in range(post_join_commits):
        await _commit(stream)

    # Drain: let the catch-up task and any jittered deliveries finish.
    for _ in range(200):
        if all(t.done() for t in stream._catchup_tasks.values()) and role2.received:  # noqa: SLF001
            break
        await asyncio.sleep(0.001)

    ctx = (
        f"seed={seed} jitter={jitter_s} pre={pre_join_commits} "
        f"post={post_join_commits} pumps={post_join_pumps}"
    )

    assert role2.started >= 1, f"joiner never got stream/start ({ctx})"
    assert role2.received, f"joiner was stranded with no audio ({ctx})"

    role2_chunks = sorted(role2.received, key=lambda c: c.timestamp_us)
    role2_ts = [c.timestamp_us for c in role2_chunks]
    role1_ts = {c.timestamp_us for c in role1.received}

    for prev, nxt in pairwise(role2_chunks):
        assert nxt.timestamp_us == prev.timestamp_us + prev.duration_us, (
            f"joiner timeline has a gap/overlap: {role2_ts} ({ctx})"
        )

    assert set(role2_ts) <= role1_ts, (
        f"joiner desynced from group: role2={role2_ts} "
        f"not a subset of role1={sorted(role1_ts)} ({ctx})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", SEEDS + LARGE_SWEEP_SEEDS)
async def test_late_join_scheduling_race(seed: int) -> None:
    """Late join stays on the group timeline under perturbed scheduling order."""
    await _run_scenario(seed)
