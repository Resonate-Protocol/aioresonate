"""Push-based audio streaming API."""

from __future__ import annotations

__all__ = ["MAIN_CHANNEL", "PushStream"]

import asyncio
import logging
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple
from uuid import UUID

from aiosendspin.server.audio import (
    AudioFormat,
    _convert_s32_to_s24,
    _get_av,
)
from aiosendspin.server.audio_transformers import TransformKey, normalize_options
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles import AudioChunk
from aiosendspin.server.roles.player.audio_transformers import PcmPassthrough
from aiosendspin.util import create_task

if TYPE_CHECKING:
    import av

    from aiosendspin.server.audio_transformers import AudioTransformer
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.clock import Clock
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles import AudioRequirements, Role

_LOGGER = logging.getLogger(__name__)

# TODO: test if still required, since I fixed double stream start messages
# Default initial delay before first audio plays (microseconds)
DEFAULT_INITIAL_DELAY_US = 250_000  # 250ms
# Pre-roll amount for catch-up encoding to absorb codec startup delay.
ENCODER_CATCHUP_WARMUP_US = 120_000
# Maximum allowed drift between transformer's internal timeline and the expected output
# timestamp before we discard the transformer's timeline. Normal codec buffering delays are
# < 100ms; this threshold catches the accumulated drift from timeline rebasing.
_TRANSFORMER_DRIFT_THRESHOLD_US = 500_000  # 500ms


def _encode_for_transform_key(
    transformer: AudioTransformer | None,
    pcm_data: bytes,
    output_ts: int,
    duration_us: int,
) -> list[tuple[bytes, int, int]]:
    """Encode PCM for a single TransformKey. Thread-safe (no shared state)."""
    if transformer is None:
        return [(pcm_data, output_ts, duration_us)]

    frames = transformer.process(pcm_data, output_ts, duration_us)
    if not frames:
        return []

    frame_dur = transformer.frame_duration_us
    base_ts = output_ts
    pending = transformer.pending_timestamp_us
    if pending is not None:
        candidate_base = pending - (len(frames) * frame_dur)
        if abs(candidate_base - output_ts) <= _TRANSFORMER_DRIFT_THRESHOLD_US:
            base_ts = candidate_base

    return [(data, base_ts + i * frame_dur, frame_dur) for i, data in enumerate(frames)]


class _ResamplerKey(NamedTuple):
    """Key for sharing resamplers: (channel_id, source_format, target PCM params).

    Resamplers convert PCM from source format to target sample rate/channels/bit_depth.
    The codec is irrelevant for resampling, so multiple target formats with different
    codecs but the same PCM parameters can share a resampler.
    """

    channel_id: UUID
    source_format: AudioFormat
    target_sample_rate: int
    target_channels: int
    target_bit_depth: int


@dataclass
class _ResamplerState:
    """Shared resampler state keyed by _ResamplerKey."""

    key: _ResamplerKey
    """Resampler key for identification."""
    resampler: av.AudioResampler
    """PyAV audio resampler."""
    source_av_format: str
    """PyAV format string for source."""
    source_av_layout: str
    """PyAV channel layout for source."""
    target_av_format: str
    """PyAV format string for target (after resampling)."""
    target_layout: str
    """PyAV channel layout for target."""
    target_wire_frame_stride: int
    """Bytes per frame for wire PCM representation."""
    target_av_frame_stride: int
    """Bytes per frame in PyAV representation."""
    needs_s32_to_s24_conversion: bool
    """True when resampler output is s32 but wire PCM is packed s24."""
    pending_timestamp_us: int | None = None
    """Timestamp of the earliest audio sample not yet emitted by this resampler."""


@dataclass(frozen=True)
class _ResampledPCM:
    """Resampler output with timestamp and sample metadata."""

    pcm_data: bytes
    output_start_ts: int
    sample_count: int
    needs_s32_to_s24_conversion: bool


def _create_resampler_state(
    key: _ResamplerKey,
    source_format: AudioFormat,
    target_format: AudioFormat,
) -> _ResamplerState:
    """Create a new resampler state. Thread-safe (no shared state)."""
    av = _get_av()

    _source_wire_bytes, source_av_format, source_layout, _source_av_bytes = (
        source_format.resolve_av_format()
    )
    target_wire_bytes, target_av_format, target_layout, target_av_bytes = (
        target_format.resolve_av_format()
    )

    resampler = av.AudioResampler(
        format=target_av_format,
        layout=target_layout,
        rate=target_format.sample_rate,
    )

    return _ResamplerState(
        key=key,
        resampler=resampler,
        source_av_format=source_av_format,
        source_av_layout=source_layout,
        target_av_format=target_av_format,
        target_layout=target_layout,
        target_wire_frame_stride=target_wire_bytes * target_format.channels,
        target_av_frame_stride=target_av_bytes * target_format.channels,
        needs_s32_to_s24_conversion=(
            target_format.bit_depth == 24 and target_av_bytes != target_wire_bytes
        ),
    )


def _resample_pcm_standalone(
    resampler_state: _ResamplerState,
    source_pcm: bytes,
    source_format: AudioFormat,
    input_timestamp_us: int,
) -> _ResampledPCM:
    """Resample PCM data to the target format.

    Thread-safe per resampler_state instance (each call should use its own state).

    Args:
        resampler_state: The resampler state to use.
        source_pcm: Source PCM bytes.
        source_format: Source audio format.
        input_timestamp_us: Timestamp for the input audio.

    Returns:
        Resampled PCM with timestamp and sample metadata.
    """
    av = _get_av()

    # Handle timestamp tracking
    if resampler_state.pending_timestamp_us is None:
        resampler_state.pending_timestamp_us = input_timestamp_us
    else:
        # Resync if timestamp drifts too far (e.g., resampler was idle)
        drift_us = abs(resampler_state.pending_timestamp_us - input_timestamp_us)
        if drift_us > 20_000:
            resampler_state.pending_timestamp_us = input_timestamp_us
            resampler_state.resampler = av.AudioResampler(
                format=resampler_state.target_av_format,
                layout=resampler_state.target_layout,
                rate=resampler_state.key.target_sample_rate,
            )

    # Calculate sample count from input
    bytes_per_sample = source_format.bit_depth // 8
    frame_stride = bytes_per_sample * source_format.channels
    sample_count = len(source_pcm) // frame_stride

    if sample_count == 0:
        return _ResampledPCM(
            pcm_data=b"",
            output_start_ts=resampler_state.pending_timestamp_us,
            sample_count=0,
            needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
        )

    # Create input frame
    frame = av.AudioFrame(
        format=resampler_state.source_av_format,
        layout=resampler_state.source_av_layout,
        samples=sample_count,
    )
    frame.sample_rate = source_format.sample_rate
    frame.planes[0].update(source_pcm)

    # Resample
    out_frames = resampler_state.resampler.resample(frame)
    out_pcm = bytearray()
    output_sample_count = 0
    for out_frame in out_frames:
        expected = resampler_state.target_av_frame_stride * out_frame.samples
        pcm_bytes = bytes(out_frame.planes[0])[:expected]
        out_pcm.extend(pcm_bytes)
        output_sample_count += out_frame.samples

    output_start_ts = resampler_state.pending_timestamp_us

    # Update pending timestamp based on output samples
    duration_us = int(output_sample_count * 1_000_000 / resampler_state.key.target_sample_rate)
    resampler_state.pending_timestamp_us += duration_us

    return _ResampledPCM(
        pcm_data=bytes(out_pcm),
        output_start_ts=output_start_ts,
        sample_count=output_sample_count,
        needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
    )


# Minimum lead time (from now) for sending catch-up audio to late joiners.
# This must be lower than DEFAULT_INITIAL_DELAY_US, otherwise a steady-state low-latency
# stream may have no chunks whose *start* timestamp is >= now + DEFAULT_INITIAL_DELAY_US.
LATE_JOINER_MIN_LEAD_US = 100_000  # 100ms


@dataclass(frozen=True)
class CachedChunk:
    """Cached chunk for late joiner catch-up."""

    timestamp_us: int
    """Start timestamp for this chunk."""
    duration_us: int
    """Duration of this chunk in microseconds."""
    payload: bytes
    """Encoded audio payload bytes (without binary header)."""
    byte_count: int
    """Size of encoded audio data (without header)."""


@dataclass(frozen=True, slots=True)
class CachedPCMChunk:
    """Raw PCM cache entry for catch-up encoding."""

    timestamp_us: int
    duration_us: int
    pcm_data: bytes
    sample_rate: int
    bit_depth: int
    channels: int


class StreamStoppedError(Exception):
    """Raised when trying to commit audio on a stopped stream."""


class PushStream:
    """
    Push-based audio streaming API.

    This class provides a push-based interface for streaming audio to players.
    Audio is prepared via prepare_audio(), then committed and sent via commit_audio().
    Late audio is handled by the connection layer (dropped if past playback time).
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        clock: Clock,
        group: SendspinGroup,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            clock: Time source used for timestamping.
            group: Group this stream belongs to.
        """
        self._loop = loop
        self._clock = clock
        self._group = group
        self._is_stopped = False
        # Pending audio per channel: channel_id -> (pcm_bytes, audio_format)
        self._channel_buffers: dict[UUID, tuple[bytes, AudioFormat]] = {}
        # Per-channel timing: channel_id -> next_chunk_start_us
        self._channel_timing: dict[UUID, int] = {}
        # Channels that have committed real audio (live or historical), not just synthetic timing.
        self._channels_with_committed_audio: set[UUID] = set()
        # Role-based streaming tracking (for hook-based flow)
        self._started_roles: set[Role] = set()
        # Inline resamplers for role-based audio delivery
        self._resamplers: dict[_ResamplerKey, _ResamplerState] = {}
        # Role-based chunk cache: TransformKey -> list of cached chunks
        self._role_chunk_cache: defaultdict[TransformKey, list[CachedChunk]] = defaultdict(list)
        # PCM chunk cache: channel_id.int -> deque of cached PCM chunks
        self._pcm_chunk_cache: dict[int, deque[CachedPCMChunk]] = {}
        # Channels with PCM caching enabled
        self._pcm_cache_enabled_channels: set[int] = {MAIN_CHANNEL.int}
        # Catch-up encoding state per TransformKey
        self._catchup_state: dict[TransformKey, Literal["catching_up", "live"]] = {}
        self._catchup_roles: dict[TransformKey, set[Role]] = {}
        self._catchup_tasks: dict[TransformKey, asyncio.Task[None]] = {}
        # TransformKey cache by (role_id, channel_id_int) - avoids rebuilding keys each frame
        self._transform_key_cache: dict[tuple[int, int], TransformKey] = {}
        # Last encoded input end timestamp per TransformKey for long-gap reset handling.
        self._transform_last_input_end_us: dict[TransformKey, int] = {}
        # Roles awaiting delayed join; excluded from live delivery until join executes.
        self._pending_join_roles: weakref.WeakSet[Role] = weakref.WeakSet()
        # Historical audio buffers: channel_id -> list of (pcm_bytes, audio_format)
        self._historical_buffers: dict[UUID, list[tuple[bytes, AudioFormat]]] = {}
        # Optional start timestamps for historical channels (set on first historical chunk).
        self._historical_start_us: dict[UUID, int] = {}

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

    @staticmethod
    def _client_in_audio_pipeline(client: SendspinClient) -> bool:
        """Whether a client should participate in transform/delivery processing."""
        # Keep warm-disconnected roles in the pipeline: if audio processing
        # stops during a temporary disconnect, we stop generating/caching role
        # outputs. On reconnect that leaves no backlog for catch-up and creates
        # an audible gap. Warm roles keep push processing running while the
        # transport is down so reconnect can resume from cached output.
        return client.is_connected or client.has_warm_disconnected_roles

    @classmethod
    def _role_in_audio_pipeline(cls, client: SendspinClient, role: Role) -> bool:
        """Whether a specific role should participate in transform/delivery processing."""
        if cls._client_in_audio_pipeline(client):
            return True
        if not client.has_cold_preinitialized_roles:
            return False
        return role.supports_preconnect_audio()

    def _get_audio_roles(self) -> list[tuple[SendspinClient, Role]]:
        """Get all roles that need audio from connected/warm/cold-opted-in clients."""
        result: list[tuple[SendspinClient, Role]] = []
        for client in self._group.clients:
            for role in client.active_roles:
                if role.get_audio_requirements() is None:
                    continue
                if role in self._pending_join_roles:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                result.append((client, role))
        return result

    def _get_cached_resampler(self, key: _ResamplerKey) -> _ResamplerState | None:
        """Get existing resampler from cache, or None if not cached."""
        return self._resamplers.get(key)

    def _cache_resampler(self, state: _ResamplerState) -> None:
        """Store a resampler state in the cache."""
        self._resamplers[state.key] = state

    def has_pending_audio(self) -> bool:
        """Return True if there is pending audio to commit."""
        return len(self._channel_buffers) > 0

    def enable_pcm_cache_for_channel(self, channel_id: UUID) -> None:
        """Enable raw PCM caching for a channel."""
        self._pcm_cache_enabled_channels.add(channel_id.int)

    def disable_pcm_cache_for_channel(self, channel_id: UUID) -> None:
        """Disable raw PCM caching for a channel."""
        channel_int = channel_id.int
        self._pcm_cache_enabled_channels.discard(channel_int)
        self._pcm_chunk_cache.pop(channel_int, None)

    def _has_pcm_cache(self, channel_id: UUID) -> bool:
        """Return True if PCM caching is enabled and cached chunks exist."""
        channel_int = channel_id.int
        if channel_int not in self._pcm_cache_enabled_channels:
            return False
        return bool(self._pcm_chunk_cache.get(channel_int))

    def get_pending_audio(self) -> dict[UUID, tuple[bytes, AudioFormat]]:
        """Return the pending audio buffers (for testing/inspection)."""
        return self._channel_buffers

    async def sleep_to_limit_buffer(self, max_buffer_us: int) -> None:
        """Sleep until the furthest-ahead active channel is at most max_buffer_us ahead of now.

        Only considers channels with active audio roles; stale channels from ungrouped
        players are ignored. Falls back to all channel timings when no roles are active
        to prevent a tight loop that starves the event loop.

        :param max_buffer_us: Maximum allowed buffer depth in microseconds.
        """
        if not self._channel_timing:
            return
        active_channels = self._get_active_audio_channels()
        active_timings = [t for ch, t in self._channel_timing.items() if ch in active_channels]
        if not active_timings:
            # Fall back to all channel timings when no audio roles are active yet
            # (e.g. client connected but handshake not complete). Without this,
            # the caller's commit loop never sleeps, starving the event loop and
            # preventing the handshake from completing.
            active_timings = list(self._channel_timing.values())
        if not active_timings:
            return
        max_timing_us = max(active_timings)
        now_us = self._clock.now_us()
        ahead_us = max_timing_us - now_us
        if ahead_us > max_buffer_us:
            await asyncio.sleep(min((ahead_us - max_buffer_us) / 1_000_000, 1.0))

    def prepare_audio(
        self,
        pcm: bytes,
        audio_format: AudioFormat,
        *,
        channel_id: UUID = MAIN_CHANNEL,
    ) -> None:
        """
        Prepare PCM audio for the next commit.

        This is a synchronous method that stores the PCM data for encoding
        during commit_audio(). Calling twice for the same channel replaces
        the previous data (does not append).

        Args:
            pcm: Raw PCM audio data.
            audio_format: Format of the PCM data.
            channel_id: Channel to prepare audio for (default: MAIN_CHANNEL).
        """
        self._channel_buffers[channel_id] = (pcm, audio_format)

    def prepare_historical_audio(
        self,
        pcm: bytes,
        audio_format: AudioFormat,
        *,
        channel_id: UUID = MAIN_CHANNEL,
        start_time_us: int | None = None,
    ) -> None:
        """Queue historical PCM audio for a new channel.

        Called multiple times to accumulate historical chunks (oldest first).
        On commit_audio(), timestamps are assigned so historical chunks play
        consecutively starting at "now + lead_time", and the live chunk
        (from prepare_audio) continues seamlessly after.

        :param pcm: Raw PCM audio data.
        :param audio_format: Format of the PCM data.
        :param channel_id: Channel to inject historical audio into.
        :param start_time_us: Optional explicit timestamp for the first historical
            chunk on this channel. If omitted, commit_audio() uses now+initial_delay.

        Raises:
            ValueError: If the channel already has active timing.
        """
        if channel_id in self._channels_with_committed_audio:
            raise ValueError(
                f"Cannot add historical audio to channel {channel_id} - "
                "channel already has active timing"
            )
        if start_time_us is not None and channel_id not in self._historical_start_us:
            self._historical_start_us[channel_id] = start_time_us
        self._historical_buffers.setdefault(channel_id, []).append((pcm, audio_format))

    def get_cached_pcm_chunks(self, channel_id: UUID = MAIN_CHANNEL) -> list[CachedPCMChunk]:
        """Retrieve cached PCM chunks for a channel.

        :param channel_id: Channel to retrieve cache from.

        Returns:
            List of CachedPCMChunk objects in chronological order.
        """
        channel_int = channel_id.int
        if channel_int not in self._pcm_chunk_cache:
            return []
        return list(self._pcm_chunk_cache[channel_int])

    def get_late_join_target_timestamp_us(
        self,
        *,
        channel_id: UUID | None = None,
        align_to_channel_tail: bool = False,
        min_lead_us: int = LATE_JOINER_MIN_LEAD_US,
    ) -> int:
        """Return a safe minimum playback timestamp for late-join replay."""
        now_us = self._clock.now_us()
        target_us = now_us + max(0, min_lead_us)
        if align_to_channel_tail and channel_id is not None and channel_id in self._channel_timing:
            # For channels that currently have no other subscribers, anchor catch-up
            # to that channel's own live tail when it is near real time. If that tail
            # drifted far ahead (e.g., reconnect with large server-side buffering),
            # use the standard near-now target to avoid long audible startup delays.
            channel_tail_us = max(now_us, self._channel_timing[channel_id])
            if channel_tail_us <= now_us + DEFAULT_INITIAL_DELAY_US:
                return channel_tail_us
        return target_us

    async def commit_audio(self, *, play_start_us: int | None = None) -> int:
        """
        Encode and send all prepared audio to players.

        This is an asynchronous method that:
        1. Encodes prepared PCM for each required format
        2. Assigns timestamps to encoded chunks
        3. Sends chunks to connected players via role hooks

        Args:
            play_start_us: If provided, use this timestamp directly for all channels
                instead of auto-calculating from clock. Useful for multi-server sync
                where all servers share a clock and need identical timestamps.

        Returns:
            The earliest play_start_us timestamp across all channels.

        Raises:
            StreamStoppedError: If the stream has been stopped.
        """
        # Check if stopped
        if self._is_stopped:
            raise StreamStoppedError("Cannot commit audio on a stopped stream")

        # Drain historical buffers
        historical = dict(self._historical_buffers)
        self._historical_buffers.clear()
        historical_start_us = dict(self._historical_start_us)
        self._historical_start_us.clear()

        # If no pending audio (live or historical), return earliest channel timing
        if not self._channel_buffers and not historical:
            now_us = self._clock.now_us()
            if not self._channel_timing:
                self._channel_timing[MAIN_CHANNEL] = now_us + DEFAULT_INITIAL_DELAY_US
            return min(self._channel_timing.values())

        # Process historical buffers first: assign timestamps and inject into caches.
        # This initializes _channel_timing for historical channels so the live chunk
        # (if any) continues seamlessly after.
        if historical:
            await self._process_historical_buffers(historical, historical_start_us)

        # Drain live channel buffers
        prepared = dict(self._channel_buffers)
        self._channel_buffers.clear()

        if not prepared:
            # Historical-only commit: cache is updated by _process_historical_buffers().
            self._prune_role_chunk_cache()
            return min(self._channel_timing.values())

        # Calculate duration for each channel and warn on misalignment
        durations_us = self._calculate_channel_durations(prepared)
        self._warn_duration_misalignment(durations_us)

        # Capture play_start_us for each channel
        channel_play_start = self._resolve_channel_play_start(
            prepared,
            play_start_us=play_start_us,
        )

        # Advance each channel's timing by its duration
        for channel_id, duration_us in durations_us.items():
            self._channel_timing[channel_id] += duration_us
            self._channels_with_committed_audio.add(channel_id)

        # Keep non-prepared active channels on the shared timeline.
        #
        # This avoids channel drift when an upstream channel (e.g., per-device DSP)
        # times out and we commit only a subset of channels for one or more cycles.
        # Those channels skip audio for this commit but should remain clock-aligned
        # when they resume.
        reference_duration_us = max(durations_us.values(), default=0)
        if reference_duration_us > 0:
            base_start_us = min(channel_play_start.values())
            for channel_id in self._get_active_audio_channels():
                if channel_id in prepared:
                    continue
                if channel_id not in self._channel_timing:
                    self._channel_timing[channel_id] = base_start_us
                self._channel_timing[channel_id] += reference_duration_us

        # Cache PCM chunks before encoding (if enabled)
        for channel_id, (pcm_bytes, fmt) in prepared.items():
            channel_int = channel_id.int
            if channel_int not in self._pcm_cache_enabled_channels:
                continue
            pcm_chunk = CachedPCMChunk(
                timestamp_us=channel_play_start[channel_id],
                duration_us=durations_us[channel_id],
                pcm_data=pcm_bytes,
                sample_rate=fmt.sample_rate,
                bit_depth=fmt.bit_depth,
                channels=fmt.channels,
            )
            self._pcm_chunk_cache.setdefault(channel_int, deque()).append(pcm_chunk)

        # Role-based audio delivery via hooks
        role_cache_results = await self._deliver_audio_to_roles(prepared, channel_play_start)
        # Merge role-based cache results into the cache
        for cache_key, chunks in role_cache_results.items():
            self._role_chunk_cache[cache_key].extend(chunks)

        # Prune old chunks from cache
        self._prune_role_chunk_cache()
        self._prune_stale_channel_timing()

        # Return earliest play_start_us
        return min(channel_play_start.values())

    def _resolve_channel_play_start(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        *,
        play_start_us: int | None,
    ) -> dict[UUID, int]:
        """Resolve play-start timestamp per prepared channel and initialize timing."""
        channel_play_start: dict[UUID, int] = {}

        if play_start_us is not None:
            # Explicit timestamp mode: use provided timestamp directly.
            for channel_id in prepared:
                channel_play_start[channel_id] = play_start_us
                if channel_id not in self._channel_timing:
                    self._channel_timing[channel_id] = play_start_us
            return channel_play_start

        # Auto-calculate mode (existing behavior).
        now_us = self._clock.now_us()
        target_min_us = now_us + DEFAULT_INITIAL_DELAY_US
        # Limit timeline sharing/rebase inputs to channels participating in this commit
        # (active subscribers + prepared payloads). This excludes stale timing entries
        # from inactive channels.
        active_or_prepared_channels = self._get_active_audio_channels() | set(prepared)
        for channel_id in prepared:
            if channel_id not in self._channel_timing:
                shared_candidates = [
                    self._channel_timing[cid]
                    for cid in active_or_prepared_channels
                    if cid in self._channel_timing
                ]
                if shared_candidates:
                    # Late-introduced channels should join the shared timeline
                    # instead of restarting from now+delay behind active channels.
                    shared_timing_us = min(shared_candidates)
                    self._channel_timing[channel_id] = max(shared_timing_us, target_min_us)
                else:
                    self._channel_timing[channel_id] = target_min_us

        # If audio production stalls (e.g., the upstream source blocks), the scheduled
        # play timeline can drift into the past. Rebase the timeline so new audio is
        # always scheduled with at least the default initial delay from "now".
        rebase_candidates = [
            self._channel_timing[cid]
            for cid in active_or_prepared_channels
            if cid in self._channel_timing
        ]
        if not rebase_candidates:
            # During reconnect/handshake edges there may be no active/prepared channels.
            # Fall back to all timings to keep existing behavior in that case.
            rebase_candidates = list(self._channel_timing.values())
        min_timing_us = min(rebase_candidates)
        if min_timing_us < target_min_us:
            shift_us = target_min_us - min_timing_us
            for channel_id in self._channel_timing:
                self._channel_timing[channel_id] += shift_us

        for channel_id in prepared:
            channel_play_start[channel_id] = self._channel_timing[channel_id]
        return channel_play_start

    async def _process_historical_buffers(
        self,
        historical: dict[UUID, list[tuple[bytes, AudioFormat]]],
        historical_start_us: dict[UUID, int] | None = None,
    ) -> None:
        """Process historical audio buffers, assigning timestamps consecutively.

        For each channel with historical data:
        1. Initialize timing from explicit start, shared timeline alignment, or default delay
        2. Assign timestamps so chunks play consecutively
        3. Cache PCM and encode for active roles
        4. Advance channel timing to end of last historical chunk

        :param historical: Channel ID -> list of (pcm, format) chunks (oldest first).
        """
        now_us = self._clock.now_us()
        min_delivery_timestamp_us = now_us + DEFAULT_INITIAL_DELAY_US

        for channel_id, chunks in historical.items():
            total_duration_us = 0
            for pcm_bytes, fmt in chunks:
                bytes_per_sample = fmt.bit_depth // 8
                frame_stride = bytes_per_sample * fmt.channels
                sample_count = len(pcm_bytes) // frame_stride
                total_duration_us += int(sample_count * 1_000_000 / fmt.sample_rate)

            if historical_start_us is not None and channel_id in historical_start_us:
                self._channel_timing[channel_id] = historical_start_us[channel_id]
            elif channel_id in self._channel_timing:
                if channel_id not in self._channels_with_committed_audio:
                    # Synthetic timing (from alignment of missing channels) should not block
                    # history injection. Backfill from current channel tail so history ends
                    # exactly at the existing shared timeline position.
                    self._channel_timing[channel_id] = max(
                        0, self._channel_timing[channel_id] - total_duration_us
                    )
            elif self._channel_timing:
                # Align injected history so it ends at the current shared timeline.
                # This avoids leaving newly injected channels permanently behind live.
                # Prefer active-channel timings so inactive stale entries do not
                # pull historical injection behind the live timeline.
                active_channels = self._get_active_audio_channels()
                anchor_candidates = [
                    self._channel_timing[cid]
                    for cid in active_channels
                    if cid in self._channel_timing
                ]
                if not anchor_candidates:
                    # Fallback preserves historical-only behavior when no roles are active.
                    anchor_candidates = list(self._channel_timing.values())
                anchor_timing_us = min(anchor_candidates)
                self._channel_timing[channel_id] = max(0, anchor_timing_us - total_duration_us)
            else:
                self._channel_timing[channel_id] = now_us + DEFAULT_INITIAL_DELAY_US

            for pcm_bytes, fmt in chunks:
                chunk_start_us = self._channel_timing[channel_id]

                # Calculate duration
                bytes_per_sample = fmt.bit_depth // 8
                frame_stride = bytes_per_sample * fmt.channels
                sample_count = len(pcm_bytes) // frame_stride
                duration_us = int(sample_count * 1_000_000 / fmt.sample_rate)

                # Cache PCM (if enabled)
                channel_int = channel_id.int
                if channel_int in self._pcm_cache_enabled_channels:
                    pcm_chunk = CachedPCMChunk(
                        timestamp_us=chunk_start_us,
                        duration_us=duration_us,
                        pcm_data=pcm_bytes,
                        sample_rate=fmt.sample_rate,
                        bit_depth=fmt.bit_depth,
                        channels=fmt.channels,
                    )
                    self._pcm_chunk_cache.setdefault(channel_int, deque()).append(pcm_chunk)

                # Advance timing even if we skip delivery for stale chunks.
                self._channel_timing[channel_id] += duration_us

                # For late-join injection, historical chunks may already be too old by the
                # time they're processed. Keep timeline/cache continuity, but don't deliver
                # chunks that would be immediately dropped as late by the connection layer.
                if chunk_start_us + duration_us <= min_delivery_timestamp_us:
                    continue

                # Encode and deliver to roles
                prepared = {channel_id: (pcm_bytes, fmt)}
                play_start = {channel_id: chunk_start_us}
                role_cache_results = await self._deliver_audio_to_roles(prepared, play_start)
                for cache_key, cached_chunks in role_cache_results.items():
                    self._role_chunk_cache[cache_key].extend(cached_chunks)
                # Yield so historical injection doesn't starve the event loop.
                await asyncio.sleep(0)
            self._channels_with_committed_audio.add(channel_id)

    def _calculate_channel_durations(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
    ) -> dict[UUID, int]:
        """Calculate duration in microseconds for each prepared channel."""
        durations: dict[UUID, int] = {}
        for channel_id, (pcm, fmt) in prepared.items():
            bytes_per_sample = fmt.bit_depth // 8
            frame_stride = bytes_per_sample * fmt.channels
            sample_count = len(pcm) // frame_stride
            duration_us = int(sample_count * 1_000_000 / fmt.sample_rate)
            durations[channel_id] = duration_us
        return durations

    def _warn_duration_misalignment(self, durations_us: dict[UUID, int]) -> None:
        """Log a warning if channel durations differ significantly."""
        if len(durations_us) <= 1:
            return

        values = list(durations_us.values())
        min_dur = min(values)
        max_dur = max(values)

        # Warn if durations differ by more than 5ms
        tolerance_us = 5000
        if max_dur - min_dur > tolerance_us:
            _LOGGER.warning(
                "Channel durations differ by %dus (tolerance: %dus)",
                max_dur - min_dur,
                tolerance_us,
            )

    def _group_roles_by_pcm_requirements(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
    ) -> dict[tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]]:
        """Group roles by their PCM requirements (channel_id, sample_rate, bit_depth, channels)."""
        # Key type: (channel_id, sample_rate, bit_depth, channels)
        roles_by_pcm: defaultdict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ] = defaultdict(list)

        for client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req is None:
                continue

            channel_id = req.channel_id or MAIN_CHANNEL
            if channel_id not in prepared:
                continue

            pcm_key = (channel_id, req.sample_rate, req.bit_depth, req.channels)
            roles_by_pcm[pcm_key].append((client, role, req))

        return roles_by_pcm

    def _get_active_audio_channels(self) -> set[UUID]:
        """Return channels currently used by connected audio roles."""
        channels: set[UUID] = set()
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req is None:
                continue
            channels.add(req.channel_id or MAIN_CHANNEL)
        return channels

    async def _resample_for_roles(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
    ) -> dict[tuple[UUID, int, int, int], _ResampledPCM]:
        """Resample PCM once per unique PCM key. Returns (channel, rate, depth, ch) -> (pcm, ts).

        Resampler state (PyAV objects) is cached and must stay on a single thread.
        Running cached resamplers across a worker pool can deadlock depending on
        thread scheduling, so resampling runs synchronously on the loop thread.
        """
        if not roles_by_pcm:
            return {}

        results: dict[tuple[UUID, int, int, int], _ResampledPCM] = {}
        for pcm_key in roles_by_pcm:
            channel_id, target_sample_rate, target_bit_depth, target_channels = pcm_key
            source_pcm, source_format = prepared[channel_id]
            input_timestamp_us = channel_play_start[channel_id]

            target_format = AudioFormat(
                sample_rate=target_sample_rate,
                bit_depth=target_bit_depth,
                channels=target_channels,
            )

            resampler_key = _ResamplerKey(
                channel_id=channel_id,
                source_format=source_format,
                target_sample_rate=target_sample_rate,
                target_channels=target_channels,
                target_bit_depth=target_bit_depth,
            )

            state = self._get_cached_resampler(resampler_key)
            if state is None:
                state = _create_resampler_state(resampler_key, source_format, target_format)
                self._cache_resampler(state)

            results[pcm_key] = _resample_pcm_standalone(
                state, source_pcm, source_format, input_timestamp_us
            )
        return results

    def _resolve_frame_duration_us(self, req: AudioRequirements) -> int:
        if req.frame_duration_us is not None:
            return req.frame_duration_us
        if req.transformer is not None:
            return req.transformer.frame_duration_us
        return 25_000

    def _build_transform_key(
        self, req: AudioRequirements, channel_id: UUID, role: Role | None = None
    ) -> TransformKey:
        # Cache by (role_id, channel_id_int); invalidated by on_role_format_changed()
        if role is not None:
            cache_key = (id(role), channel_id.int)
            cached = self._transform_key_cache.get(cache_key)
            if cached is not None:
                return cached

        transformer_type = type(req.transformer) if req.transformer is not None else type(None)
        tkey = TransformKey(
            channel_id=channel_id.int,  # Use int for faster hashing
            transformer_type=transformer_type,
            sample_rate=req.sample_rate,
            bit_depth=req.bit_depth,
            channels=req.channels,
            frame_duration_us=self._resolve_frame_duration_us(req),
            options=normalize_options(req.transform_options),
        )

        if role is not None:
            self._transform_key_cache[cache_key] = tkey

        return tkey

    def _encode_transform_for_key(
        self,
        tkey: TransformKey,
        transformer: AudioTransformer | None,
        pcm_data: bytes,
        output_ts: int,
        duration_us: int,
    ) -> list[tuple[bytes, int, int]]:
        """Encode PCM while resetting transformer state after long production gaps."""
        last_input_end_us = self._transform_last_input_end_us.get(tkey)
        if (
            transformer is not None
            and last_input_end_us is not None
            and output_ts - last_input_end_us > 1_500_000
        ):
            transformer.reset()

        # Reset transformer if its internal timeline has drifted too far from the
        # expected output position. This catches gradual drift from timeline rebasing
        # (when audio production is slower than real-time, each commit's rebase shifts
        # timestamps forward, but the transformer's internal timeline only advances by
        # the actual audio duration).
        if (
            transformer is not None
            and transformer.pending_timestamp_us is not None
            and abs(transformer.pending_timestamp_us - (output_ts + duration_us))
            > _TRANSFORMER_DRIFT_THRESHOLD_US
        ):
            transformer.reset()

        encoded = _encode_for_transform_key(
            transformer,
            pcm_data,
            output_ts,
            duration_us,
        )
        self._transform_last_input_end_us[tkey] = output_ts + duration_us
        return encoded

    async def _transform_and_deliver(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        resampled_pcm: dict[tuple[UUID, int, int, int], _ResampledPCM],
    ) -> dict[TransformKey, list[CachedChunk]]:
        """Transform PCM, deliver live chunks to roles, and return cache results.

        Encoding is parallelized across unique TransformKeys via a thread pool.
        """
        # Collect unique encoding tasks: tkey -> (transformer, pcm_data, output_ts, duration_us)
        encode_tasks: dict[TransformKey, tuple[AudioTransformer | None, bytes, int, int]] = {}
        roles_by_transform: defaultdict[TransformKey, list[Role]] = defaultdict(list)

        for pcm_key, roles_list in roles_by_pcm.items():
            channel_id, rate, depth, _channels = pcm_key
            resampled = resampled_pcm[pcm_key]
            pcm_data = resampled.pcm_data
            output_ts = resampled.output_start_ts
            duration_us = int(resampled.sample_count * 1_000_000 / rate) if rate > 0 else 0

            grouped_by_key: defaultdict[
                TransformKey, list[tuple[SendspinClient, Role, AudioRequirements]]
            ] = defaultdict(list)
            for client, role, req in roles_list:
                tkey = self._build_transform_key(req, channel_id, role)
                grouped_by_key[tkey].append((client, role, req))

            for tkey, grouped in grouped_by_key.items():
                roles_by_transform[tkey].extend(role for _client, role, _req in grouped)
                if self._catchup_state.get(tkey) == "catching_up":
                    continue
                if tkey in encode_tasks:
                    continue
                transformer = grouped[0][2].transformer
                transformed_pcm = pcm_data
                if (
                    resampled.needs_s32_to_s24_conversion
                    and depth == 24
                    and isinstance(transformer, PcmPassthrough)
                ):
                    transformed_pcm = _convert_s32_to_s24(transformed_pcm)
                encode_tasks[tkey] = (transformer, transformed_pcm, output_ts, duration_us)

        # TransformKey -> list of (data, timestamp_us, duration_us)
        transformed: dict[TransformKey, list[tuple[bytes, int, int]]] = {}

        if encode_tasks:
            for processed, (tkey, (transformer, pcm_data, output_ts, duration_us)) in enumerate(
                encode_tasks.items(), start=1
            ):
                transformed[tkey] = self._encode_transform_for_key(
                    tkey,
                    transformer,
                    pcm_data,
                    output_ts,
                    duration_us,
                )
                if processed % 2 == 0:
                    # Keep the loop responsive during large multi-role commits.
                    await asyncio.sleep(0)

        cache_results: defaultdict[TransformKey, list[CachedChunk]] = defaultdict(list)

        for tkey, frame_list in transformed.items():
            cached_for_key: list[CachedChunk] = []
            for data, ts, dur in frame_list:
                cached = CachedChunk(
                    timestamp_us=ts, duration_us=dur, payload=data, byte_count=len(data)
                )
                cached_for_key.append(cached)
            if not cached_for_key:
                continue
            cache_results[tkey].extend(cached_for_key)

            # Deliver live chunks directly; connection layer enforces late-drop/backpressure.
            roles = roles_by_transform.get(tkey, [])
            for role in roles:
                self._ensure_role_started(role)
                for cached_chunk in cached_for_key:
                    role.on_audio_chunk(
                        AudioChunk(
                            data=cached_chunk.payload,
                            timestamp_us=cached_chunk.timestamp_us,
                            duration_us=cached_chunk.duration_us,
                            byte_count=cached_chunk.byte_count,
                        )
                    )

        return cache_results

    async def _deliver_audio_to_roles(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
    ) -> dict[TransformKey, list[CachedChunk]]:
        """
        Deliver audio to roles using the hook-based flow.

        This method:
        1. Groups roles by unique PCM requirements
        2. Resamples source PCM to each unique target format
        3. Transforms and delivers via role.on_audio_chunk()

        Returns:
            Dict of TransformKey -> list of CachedChunk for late joiners.
        """
        roles_by_pcm = self._group_roles_by_pcm_requirements(prepared)
        if not roles_by_pcm:
            return {}

        resampled = await self._resample_for_roles(roles_by_pcm, prepared, channel_play_start)
        return await self._transform_and_deliver(roles_by_pcm, resampled)

    def _prune_role_chunk_cache(self) -> None:
        """Remove old chunks from the role-based cache."""
        now_us = self._clock.now_us()

        for key in list(self._role_chunk_cache.keys()):
            chunks = self._role_chunk_cache[key]
            prune_count = 0
            for chunk in chunks:
                if chunk.timestamp_us + chunk.duration_us > now_us:
                    break
                prune_count += 1

            if prune_count:
                self._role_chunk_cache[key] = chunks[prune_count:]

            if not self._role_chunk_cache[key]:
                del self._role_chunk_cache[key]

        for channel_int in list(self._pcm_chunk_cache.keys()):
            pcm_chunks = self._pcm_chunk_cache[channel_int]
            while pcm_chunks and (pcm_chunks[0].timestamp_us + pcm_chunks[0].duration_us <= now_us):
                pcm_chunks.popleft()
            if not pcm_chunks:
                self._pcm_chunk_cache.pop(channel_int, None)

    def _prune_stale_channel_timing(self) -> None:
        """Remove _channel_timing entries for channels with no active roles or pending audio."""
        active = self._get_active_audio_channels()
        for ch in list(self._channel_timing):
            if ch in active:
                continue
            if ch in self._channel_buffers:
                continue
            if ch in self._historical_buffers:
                continue
            if ch in self._channels_with_committed_audio:
                # Keep timing for channels that have received real audio so the
                # timeline continues to advance. Without this, timing is reset
                # on every commit when no roles are active, preventing
                # sleep_to_limit_buffer from throttling the commit loop.
                continue
            del self._channel_timing[ch]
            self._channels_with_committed_audio.discard(ch)

    def _ensure_role_started(self, role: Role) -> None:
        if role in self._started_roles:
            return
        role.on_stream_start()
        self._started_roles.add(role)

    def _send_cached_chunks_to_role(
        self,
        role: Role,
        cached_chunks: list[CachedChunk],
        now_us: int,
    ) -> None:
        """Send cached chunks to a role, skipping chunks that are already late."""
        skipped_late = 0
        for cached_chunk in cached_chunks:
            if cached_chunk.timestamp_us + cached_chunk.duration_us <= now_us:
                skipped_late += 1
                continue

            self._ensure_role_started(role)

            chunk = AudioChunk(
                data=cached_chunk.payload,
                timestamp_us=cached_chunk.timestamp_us,
                duration_us=cached_chunk.duration_us,
                byte_count=cached_chunk.byte_count,
            )
            role.on_audio_chunk(chunk)

        if skipped_late > 0:
            _LOGGER.debug(
                "Skipped %s late cached chunk(s) for role %s (ts < now_us=%s)",
                skipped_late,
                role.role_family,
                now_us,
            )

    def on_role_leave(self, role: Role) -> None:
        """Remove role-specific state so re-joins get fresh stream/start."""
        self._started_roles.discard(role)
        self._pending_join_roles.discard(role)
        req = role.get_audio_requirements()
        if req is not None:
            channel_id = req.channel_id or MAIN_CHANNEL
            tkey = self._build_transform_key(req, channel_id, role)
            if not self._other_roles_use_transform_key(tkey, role):
                self._role_chunk_cache.pop(tkey, None)
                self._transform_last_input_end_us.pop(tkey, None)
                if req.transformer is not None:
                    req.transformer.reset()
        for tkey in list(self._catchup_roles.keys()):
            roles = self._catchup_roles[tkey]
            roles.discard(role)
            if not roles:
                self._catchup_roles.pop(tkey, None)
                self._catchup_state.pop(tkey, None)
                task = self._catchup_tasks.pop(tkey, None)
                if task is not None:
                    task.cancel()
        # Drop cached transform keys for this role to avoid stale lookups.
        role_id = id(role)
        stale_transform_keys: set[TransformKey] = set()
        for cache_key in list(self._transform_key_cache.keys()):
            if cache_key[0] == role_id:
                stale_transform_keys.add(self._transform_key_cache[cache_key])
                self._transform_key_cache.pop(cache_key, None)
        for stale_tkey in stale_transform_keys:
            self._transform_last_input_end_us.pop(stale_tkey, None)

    def on_role_format_changed(self, role: Role) -> None:
        """Invalidate caches after a role's audio format changed mid-stream.

        Unlike on_role_leave(), this does NOT touch _started_roles or epoch.
        The role stays active; only stale caches are cleared so the next
        commit_audio() picks up the new AudioRequirements.
        """
        # Invalidate transform key cache entries for this role
        role_id = id(role)
        stale_transform_keys: set[TransformKey] = set()
        for cache_key in list(self._transform_key_cache.keys()):
            if cache_key[0] == role_id:
                stale_transform_keys.add(self._transform_key_cache[cache_key])
                self._transform_key_cache.pop(cache_key, None)
        for stale_tkey in stale_transform_keys:
            self._transform_last_input_end_us.pop(stale_tkey, None)

        # Clean up any catchup state referencing this role
        for tkey in list(self._catchup_roles.keys()):
            roles = self._catchup_roles[tkey]
            roles.discard(role)
            if not roles:
                self._catchup_roles.pop(tkey, None)
                self._catchup_state.pop(tkey, None)
                task = self._catchup_tasks.pop(tkey, None)
                if task is not None:
                    task.cancel()

    def has_cached_chunks(self) -> bool:
        """Return True if there are cached chunks for late joiners."""
        return any(len(chunks) > 0 for chunks in self._role_chunk_cache.values())

    def on_role_join(self, role: Role) -> None:
        """
        Handle late joiner catch-up via hooks.

        Uses the role-based chunk cache to deliver cached audio to a role
        that just joined.

        Args:
            role: The role that joined.
        """
        # Join immediately so replay anchors to the current shared timeline.
        # Deferring by wall-clock time can desynchronize grouped players.
        self._do_role_join(role)

    def _do_role_join(self, role: Role) -> None:
        """Execute role join with cached chunk replay."""
        self._pending_join_roles.discard(role)
        # A rejoining role (e.g. warm reconnect) must receive on_stream_start()
        # again so the new transport gets stream/start before any audio chunks.
        self._started_roles.discard(role)
        if self._is_stopped:
            return
        req = role.get_audio_requirements()
        if req is None:
            return

        channel_id = req.channel_id or MAIN_CHANNEL
        self._rebase_first_join_channel_timing(channel_id, role)

        # Get cached chunks for this transformer from the role-based cache
        cache_key = self._build_transform_key(req, channel_id, role)
        cached = self._role_chunk_cache.get(cache_key, [])

        if not cached:
            if cache_key in self._catchup_state:
                self._catchup_roles.setdefault(cache_key, set()).add(role)
                return

            if self._has_pcm_cache(channel_id) and not self._other_roles_use_transform_key(
                cache_key, role
            ):
                channel_pcm_cache = self._pcm_chunk_cache.get(channel_id.int)
                if channel_pcm_cache:
                    late_join_target_us = self.get_late_join_target_timestamp_us(
                        channel_id=channel_id,
                        align_to_channel_tail=(channel_id != MAIN_CHANNEL),
                    )
                    latest_cached_end_us = (
                        channel_pcm_cache[-1].timestamp_us + channel_pcm_cache[-1].duration_us
                    )
                    if latest_cached_end_us <= late_join_target_us:
                        self._rebase_far_ahead_join_tail(channel_id, role)
                        if self._channel_timing:
                            self._ensure_role_started(role)
                        return

                self._catchup_state[cache_key] = "catching_up"
                self._catchup_roles[cache_key] = {role}
                self._catchup_tasks[cache_key] = create_task(
                    self._start_catchup_encoding(role, req, channel_id, cache_key)
                )
                return

            if self._channel_timing:
                self._rebase_far_ahead_join_tail(channel_id, role)
                self._ensure_role_started(role)
            return

        now_us = self._clock.now_us()
        min_timestamp_us = self.get_late_join_target_timestamp_us(
            channel_id=channel_id,
            align_to_channel_tail=False,
        )

        start_index = 0
        for chunk in cached:
            if chunk.timestamp_us + chunk.duration_us > min_timestamp_us:
                break
            start_index += 1

        if start_index >= len(cached):
            if self._channel_timing:
                self._ensure_role_started(role)
            return

        if _LOGGER.isEnabledFor(logging.DEBUG):
            first_ts = cached[start_index].timestamp_us
            last_ts = cached[-1].timestamp_us
            _LOGGER.debug(
                "Late join catch-up via role hook: chunks=%s ts_range=%s..%s",
                len(cached) - start_index,
                first_ts,
                last_ts,
            )

        self._send_cached_chunks_to_role(role, cached[start_index:], now_us)

    def _other_roles_use_transform_key(self, cache_key: TransformKey, exclude_role: Role) -> bool:
        """Check if any other active pipeline role uses the same TransformKey."""
        for client in self._group.clients:
            for role in client.active_roles:
                if role is exclude_role:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                req = role.get_audio_requirements()
                if req is None:
                    continue
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                if tkey == cache_key:
                    return True
        return False

    def _channel_has_other_audio_roles(self, channel_id: UUID, exclude_role: Role) -> bool:
        """Check whether any other active pipeline role is subscribed to the channel."""
        for client in self._group.clients:
            for role in client.active_roles:
                if role is exclude_role:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                req = role.get_audio_requirements()
                if req is None:
                    continue
                role_channel_id = req.channel_id or MAIN_CHANNEL
                if role_channel_id == channel_id:
                    return True
        return False

    def _rebase_first_join_channel_timing(self, channel_id: UUID, joining_role: Role) -> None:
        """Rebase stale channel timing to the shared timeline for first joiners."""
        if channel_id not in self._channel_timing or not self._channel_timing:
            return
        if self._channel_has_other_audio_roles(channel_id, joining_role):
            return
        other_channel_timings = [
            timing_us for cid, timing_us in self._channel_timing.items() if cid != channel_id
        ]
        if not other_channel_timings:
            return
        reference_timing_us = min(other_channel_timings)
        self._channel_timing[channel_id] = max(
            self._channel_timing[channel_id], reference_timing_us
        )

    def _rebase_far_ahead_join_tail(self, channel_id: UUID, joining_role: Role) -> None:
        """Clamp far-ahead solo-channel timing so a rejoin can resume promptly."""
        if channel_id not in self._channel_timing:
            return
        if self._channel_has_other_audio_roles(channel_id, joining_role):
            return
        now_us = self._clock.now_us()
        max_resume_start_us = now_us + DEFAULT_INITIAL_DELAY_US
        self._channel_timing[channel_id] = min(
            self._channel_timing[channel_id], max_resume_start_us
        )

    def _encode_pcm_sequence(
        self,
        pcm_chunks: list[CachedPCMChunk],
        encoder: AudioTransformer | None,
        req: AudioRequirements,
        channel_id: UUID,
    ) -> list[CachedChunk]:
        """Resample PCM chunks to the target format and encode them sequentially."""
        tkey = self._build_transform_key(req, channel_id)
        target_format = AudioFormat(
            sample_rate=req.sample_rate,
            bit_depth=req.bit_depth,
            channels=req.channels,
        )
        cached: list[CachedChunk] = []
        resampler_state: _ResamplerState | None = None
        resampler_source_format: AudioFormat | None = None

        for chunk in pcm_chunks:
            source_format = AudioFormat(
                sample_rate=chunk.sample_rate,
                bit_depth=chunk.bit_depth,
                channels=chunk.channels,
            )
            if (
                resampler_state is None
                or resampler_source_format is None
                or resampler_source_format != source_format
            ):
                resampler_key = _ResamplerKey(
                    channel_id=channel_id,
                    source_format=source_format,
                    target_sample_rate=req.sample_rate,
                    target_channels=req.channels,
                    target_bit_depth=req.bit_depth,
                )
                resampler_state = _create_resampler_state(
                    resampler_key,
                    source_format,
                    target_format,
                )
                resampler_source_format = source_format

            resampled = _resample_pcm_standalone(
                resampler_state,
                chunk.pcm_data,
                source_format,
                chunk.timestamp_us,
            )
            if not resampled.pcm_data:
                continue

            resampled_pcm = resampled.pcm_data
            if (
                resampled.needs_s32_to_s24_conversion
                and req.bit_depth == 24
                and isinstance(encoder, PcmPassthrough)
            ):
                resampled_pcm = _convert_s32_to_s24(resampled_pcm)
            duration_us = (
                int(resampled.sample_count * 1_000_000 / req.sample_rate)
                if resampled.sample_count > 0
                else 0
            )

            encoded_frames = self._encode_transform_for_key(
                tkey,
                encoder,
                resampled_pcm,
                resampled.output_start_ts,
                duration_us,
            )
            for data, ts, dur in encoded_frames:
                cached.append(
                    CachedChunk(
                        timestamp_us=ts,
                        duration_us=dur,
                        payload=data,
                        byte_count=len(data),
                    )
                )

        return cached

    async def _encode_catchup_sequence(
        self,
        pcm_chunks: list[CachedPCMChunk],
        encoder: AudioTransformer | None,
        req: AudioRequirements,
        channel_id: UUID,
    ) -> list[CachedChunk]:
        return self._encode_pcm_sequence(
            pcm_chunks,
            encoder,
            req,
            channel_id,
        )

    async def _start_catchup_encoding(
        self,
        role: Role,
        req: AudioRequirements,
        channel_id: UUID,
        cache_key: TransformKey,
    ) -> None:
        """Start catch-up encoding from PCM cache for a new TransformKey."""
        channel_int = channel_id.int
        encoder = req.transformer

        try:
            if encoder is not None:
                encoder.reset()
            self._transform_last_input_end_us.pop(cache_key, None)
            pcm_chunks = list(self._pcm_chunk_cache.get(channel_int, []))
            align_to_channel_tail = channel_id != MAIN_CHANNEL
            target_ts = self.get_late_join_target_timestamp_us(
                channel_id=channel_id,
                align_to_channel_tail=align_to_channel_tail,
            )
            encode_start_ts = target_ts
            if encoder is not None and align_to_channel_tail:
                encode_start_ts = max(0, target_ts - ENCODER_CATCHUP_WARMUP_US)
            eligible = [
                chunk
                for chunk in pcm_chunks
                if chunk.timestamp_us + chunk.duration_us > encode_start_ts
            ]

            if not eligible:
                if self._channel_timing:
                    for r in self._catchup_roles.get(cache_key, {role}):
                        self._ensure_role_started(r)
                return

            encoded = await self._encode_catchup_sequence(
                eligible,
                encoder,
                req,
                channel_id,
            )

            if encoded:
                self._role_chunk_cache[cache_key].extend(encoded)

            last_encoded_end_us = (
                encoded[-1].timestamp_us + encoded[-1].duration_us if encoded else 0
            )
            # Track source PCM progress separately from encoded progress. Some
            # codecs buffer input and may emit no packets for a given chunk.
            last_source_end_us = eligible[-1].timestamp_us + eligible[-1].duration_us
            idle_loops = 0
            max_idle_loops = 50  # ~500ms at 10ms sleep

            while last_encoded_end_us < target_ts:
                await asyncio.sleep(0)

                new_pcm = [
                    chunk
                    for chunk in self._pcm_chunk_cache.get(channel_int, [])
                    if chunk.timestamp_us >= last_source_end_us
                ]
                if not new_pcm:
                    idle_loops += 1
                    if idle_loops >= max_idle_loops:
                        _LOGGER.debug(
                            "Catch-up idle timeout for %s (encoded_end=%s target=%s)",
                            cache_key,
                            last_encoded_end_us,
                            target_ts,
                        )
                        break
                    await asyncio.sleep(0.01)
                    continue
                idle_loops = 0
                last_source_end_us = new_pcm[-1].timestamp_us + new_pcm[-1].duration_us

                new_encoded = await self._encode_catchup_sequence(
                    new_pcm,
                    encoder,
                    req,
                    channel_id,
                )

                if new_encoded:
                    self._role_chunk_cache[cache_key].extend(new_encoded)
                    last_encoded_end_us = new_encoded[-1].timestamp_us + new_encoded[-1].duration_us

            now_us = self._clock.now_us()
            encoded_cache = self._role_chunk_cache.get(cache_key, [])
            for r in self._catchup_roles.get(cache_key, {role}):
                self._send_cached_chunks_to_role(r, encoded_cache, now_us)

            self._catchup_state[cache_key] = "live"
        finally:
            if self._catchup_state.get(cache_key) != "live":
                self._catchup_state.pop(cache_key, None)
                self._catchup_roles.pop(cache_key, None)
            self._catchup_tasks.pop(cache_key, None)

    def _cancel_catchup_tasks(self) -> None:
        for task in self._catchup_tasks.values():
            task.cancel()
        self._catchup_tasks.clear()
        self._catchup_state.clear()
        self._catchup_roles.clear()

    def stop(self) -> None:
        """
        Stop only this PushStream transport.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        Flushes remaining audio from transformers, then sends stream/end message
        to all roles via hooks.

        This does not change the owning group's logical playback state.
        Use this when you are about to immediately start another stream and
        want clients to remain in PLAYING state during the transition.
        Call group.stop() to stop transport and also set playback state to STOPPED.
        """
        if self._is_stopped:
            return
        self._is_stopped = True

        # Flush remaining audio from transformers and reset them
        roles_by_transform: defaultdict[TransformKey, list[Role]] = defaultdict(list)
        transformers_by_key: dict[TransformKey, AudioTransformer] = {}
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req and req.transformer:
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                roles_by_transform[tkey].append(role)
                transformers_by_key.setdefault(tkey, req.transformer)

        for tkey, transformer in transformers_by_key.items():
            final_frames = transformer.flush()
            if final_frames:
                frame_duration_us = transformer.frame_duration_us
                for frame_data in final_frames:
                    chunk = AudioChunk(
                        data=frame_data,
                        timestamp_us=0,  # Timestamp doesn't matter at stream end
                        duration_us=frame_duration_us,
                        byte_count=len(frame_data),
                    )
                    for role in roles_by_transform.get(tkey, []):
                        role.on_audio_chunk(chunk)
            transformer.reset()

        # Send stream/end to all roles with audio requirements via hooks
        for _client, role in self._get_audio_roles():
            role.on_stream_end()

        # Clear role tracking state
        self._started_roles.clear()
        self._pending_join_roles.clear()
        self._cancel_catchup_tasks()
        self._pcm_chunk_cache.clear()
        self._historical_buffers.clear()
        self._historical_start_us.clear()
        self._transform_last_input_end_us.clear()
        self._channels_with_committed_audio.clear()

    def clear(self) -> None:
        """
        Clear all pending audio and reset timing.

        This is used for seek operations where buffered audio is discarded.
        Sends stream/clear to all roles via hooks.
        """
        # Clear pending audio
        self._channel_buffers.clear()
        self._historical_buffers.clear()
        self._historical_start_us.clear()

        # Reset per-channel timing
        self._channel_timing.clear()
        self._channels_with_committed_audio.clear()

        # Clear chunk cache
        self._role_chunk_cache.clear()
        self._pending_join_roles.clear()
        self._pcm_chunk_cache.clear()
        self._transform_last_input_end_us.clear()
        self._cancel_catchup_tasks()

        # Reset inline resamplers
        self._resamplers.clear()

        # Reset transformers so they don't carry stale timestamp state
        reset_transformers: dict[TransformKey, AudioTransformer] = {}
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req and req.transformer:
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                reset_transformers.setdefault(tkey, req.transformer)

        for transformer in reset_transformers.values():
            transformer.reset()

        # Clear role tracking state
        self._started_roles.clear()

        # Send stream/clear to all roles with audio requirements via hooks
        for _client, role in self._get_audio_roles():
            role.on_stream_clear()
