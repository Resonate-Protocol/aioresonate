"""Push-based audio streaming API."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple
from uuid import UUID

from aiosendspin.server.audio import AudioFormat, _get_av, _resolve_audio_format
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles import AudioChunk
from aiosendspin.server.transform_keys import TransformKey, normalize_options
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


# Thread pool for parallel encoding
class _ExecutorHolder:
    """Holds the shared encoding thread pool (avoids global statement)."""

    executor: ThreadPoolExecutor | None = None
    lock = threading.Lock()


def _get_encode_executor() -> ThreadPoolExecutor:
    """Get or create the shared encoding thread pool."""
    if _ExecutorHolder.executor is None:
        with _ExecutorHolder.lock:
            if _ExecutorHolder.executor is None:
                _ExecutorHolder.executor = ThreadPoolExecutor(
                    max_workers=min(4, (os.cpu_count() or 1)),
                    thread_name_prefix="sendspin-encode-",
                )
    return _ExecutorHolder.executor


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
    if hasattr(transformer, "pending_timestamp_us"):
        pending = transformer.pending_timestamp_us
        if pending is not None:
            base_ts = pending - (len(frames) * frame_dur)

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
    target_frame_stride: int
    """Bytes per frame in target format."""
    pending_timestamp_us: int | None = None
    """Timestamp of the earliest audio sample not yet emitted by this resampler."""


def _create_resampler_state(
    key: _ResamplerKey,
    source_format: AudioFormat,
    target_format: AudioFormat,
) -> _ResamplerState:
    """Create a new resampler state. Thread-safe (no shared state)."""
    av = _get_av()

    _source_bytes_per_sample, source_av_format, source_layout = _resolve_audio_format(source_format)
    target_bytes_per_sample, target_av_format, target_layout = _resolve_audio_format(target_format)

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
        target_frame_stride=target_bytes_per_sample * target_format.channels,
    )


def _resample_pcm_standalone(
    resampler_state: _ResamplerState,
    source_pcm: bytes,
    source_format: AudioFormat,
    input_timestamp_us: int,
) -> tuple[bytes, int]:
    """Resample PCM data to the target format.

    Thread-safe per resampler_state instance (each call should use its own state).

    Args:
        resampler_state: The resampler state to use.
        source_pcm: Source PCM bytes.
        source_format: Source audio format.
        input_timestamp_us: Timestamp for the input audio.

    Returns:
        Tuple of (resampled_pcm_bytes, output_start_timestamp_us).
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
        return b"", resampler_state.pending_timestamp_us

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
    for out_frame in out_frames:
        expected = resampler_state.target_frame_stride * out_frame.samples
        pcm_bytes = bytes(out_frame.planes[0])[:expected]
        out_pcm.extend(pcm_bytes)

    output_start_ts = resampler_state.pending_timestamp_us

    # Update pending timestamp based on output samples
    output_sample_count = len(out_pcm) // resampler_state.target_frame_stride
    duration_us = int(output_sample_count * 1_000_000 / resampler_state.key.target_sample_rate)
    resampler_state.pending_timestamp_us += duration_us

    return bytes(out_pcm), output_start_ts


# Minimum lead time (from now) for sending catch-up audio to late joiners.
# This must be lower than DEFAULT_INITIAL_DELAY_US, otherwise a steady-state low-latency
# stream may have no chunks whose *start* timestamp is >= now + DEFAULT_INITIAL_DELAY_US.
LATE_JOINER_MIN_LEAD_US = 100_000  # 100ms

# Max chunks to pump per role before yielding to the event loop
_PUMP_BATCH_SIZE = 20


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
        # Per-role chunk cursor for cache delivery.
        self._role_chunk_cursors: weakref.WeakKeyDictionary[Role, dict[TransformKey, int]] = (
            weakref.WeakKeyDictionary()
        )
        self._cache_pump_handle: asyncio.Handle | None = None

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

    def _get_audio_roles(self) -> list[tuple[SendspinClient, Role]]:
        """Get all roles that need audio from connected clients."""
        result: list[tuple[SendspinClient, Role]] = []
        for client in self._group.clients:
            if not client.is_connected:
                continue
            result.extend(
                (client, role)
                for role in client.active_roles
                if role.get_audio_requirements() is not None
            )
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

        # If no pending audio, return earliest channel timing (or initialize)
        if not self._channel_buffers:
            now_us = self._clock.now_us()
            if not self._channel_timing:
                # Initialize MAIN_CHANNEL timing if nothing exists
                self._channel_timing[MAIN_CHANNEL] = now_us + DEFAULT_INITIAL_DELAY_US
            return min(self._channel_timing.values())

        # Drain channel buffers
        prepared = dict(self._channel_buffers)
        self._channel_buffers.clear()

        # Calculate duration for each channel and warn on misalignment
        durations_us = self._calculate_channel_durations(prepared)
        self._warn_duration_misalignment(durations_us)

        # Capture play_start_us for each channel
        channel_play_start: dict[UUID, int] = {}

        if play_start_us is not None:
            # Explicit timestamp mode: use provided timestamp directly
            for channel_id in prepared:
                channel_play_start[channel_id] = play_start_us
                # Initialize timing if not yet set
                if channel_id not in self._channel_timing:
                    self._channel_timing[channel_id] = play_start_us
        else:
            # Auto-calculate mode (existing behavior)
            now_us = self._clock.now_us()
            for channel_id in prepared:
                if channel_id not in self._channel_timing:
                    self._channel_timing[channel_id] = now_us + DEFAULT_INITIAL_DELAY_US

            # If audio production stalls (e.g., the upstream source blocks), the scheduled
            # play timeline can drift into the past. Rebase the timeline so new audio is
            # always scheduled with at least the default initial delay from "now".
            min_timing_us = min(self._channel_timing.values())
            target_min_us = now_us + DEFAULT_INITIAL_DELAY_US
            if min_timing_us < target_min_us:
                shift_us = target_min_us - min_timing_us
                for channel_id in self._channel_timing:
                    self._channel_timing[channel_id] += shift_us

            for channel_id in prepared:
                channel_play_start[channel_id] = self._channel_timing[channel_id]

        # Advance each channel's timing by its duration
        for channel_id, duration_us in durations_us.items():
            self._channel_timing[channel_id] += duration_us

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

        # Deliver cached chunks to roles up to their buffer capacity
        self._trigger_cache_pump()

        # Return earliest play_start_us
        return min(channel_play_start.values())

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

    async def _resample_for_roles(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
    ) -> dict[tuple[UUID, int, int, int], tuple[bytes, int]]:
        """Resample PCM once per unique PCM key. Returns (channel, rate, depth, ch) -> (pcm, ts).

        Both resampler creation and resampling run in the thread pool to avoid
        blocking the event loop.
        """
        if not roles_by_pcm:
            return {}

        executor = _get_encode_executor()
        running_loop = asyncio.get_running_loop()

        # Collect info for each pcm_key
        pcm_key_info: dict[
            tuple[UUID, int, int, int],
            tuple[_ResamplerKey, AudioFormat, AudioFormat, bytes, int],
        ] = {}

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

            pcm_key_info[pcm_key] = (
                resampler_key,
                source_format,
                target_format,
                source_pcm,
                input_timestamp_us,
            )

        # Separate cached vs new resamplers
        cached_states: dict[tuple[UUID, int, int, int], _ResamplerState] = {}
        keys_needing_creation: list[
            tuple[tuple[UUID, int, int, int], _ResamplerKey, AudioFormat, AudioFormat]
        ] = []

        for pcm_key, (rkey, src_fmt, tgt_fmt, _pcm, _ts) in pcm_key_info.items():
            cached = self._get_cached_resampler(rkey)
            if cached is not None:
                cached_states[pcm_key] = cached
            else:
                keys_needing_creation.append((pcm_key, rkey, src_fmt, tgt_fmt))

        # Create new resamplers in thread pool
        if keys_needing_creation:

            async def create_resampler(
                pcm_key: tuple[UUID, int, int, int],
                rkey: _ResamplerKey,
                src_fmt: AudioFormat,
                tgt_fmt: AudioFormat,
            ) -> tuple[tuple[UUID, int, int, int], _ResamplerState]:
                state = await running_loop.run_in_executor(
                    executor, _create_resampler_state, rkey, src_fmt, tgt_fmt
                )
                return (pcm_key, state)

            creation_tasks = [
                create_resampler(pcm_key, rkey, src_fmt, tgt_fmt)
                for pcm_key, rkey, src_fmt, tgt_fmt in keys_needing_creation
            ]
            creation_results = await asyncio.gather(*creation_tasks)

            # Cache newly created resamplers
            for pcm_key, state in creation_results:
                self._cache_resampler(state)
                cached_states[pcm_key] = state

        # Now run resampling in thread pool
        async def run_resample(
            pcm_key: tuple[UUID, int, int, int],
            resampler_state: _ResamplerState,
            source_pcm: bytes,
            source_format: AudioFormat,
            input_timestamp_us: int,
        ) -> tuple[tuple[UUID, int, int, int], tuple[bytes, int]]:
            result = await running_loop.run_in_executor(
                executor,
                _resample_pcm_standalone,
                resampler_state,
                source_pcm,
                source_format,
                input_timestamp_us,
            )
            return (pcm_key, result)

        resample_tasks = [
            run_resample(
                pcm_key,
                cached_states[pcm_key],
                pcm_key_info[pcm_key][3],  # source_pcm
                pcm_key_info[pcm_key][1],  # source_format
                pcm_key_info[pcm_key][4],  # input_timestamp_us
            )
            for pcm_key in pcm_key_info
        ]
        results = await asyncio.gather(*resample_tasks)
        return dict(results)

    def _resolve_frame_duration_us(self, req: AudioRequirements) -> int:
        if req.frame_duration_us is not None:
            return req.frame_duration_us
        if req.transformer is not None:
            return req.transformer.frame_duration_us
        return 25_000

    def _build_transform_key(
        self, req: AudioRequirements, channel_id: UUID, role: Role | None = None
    ) -> TransformKey:
        # Cache by (role_id, channel_id_int) since AudioRequirements don't change during streaming
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

    async def _transform_and_deliver(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        resampled_pcm: dict[tuple[UUID, int, int, int], tuple[bytes, int]],
    ) -> dict[TransformKey, list[CachedChunk]]:
        """Transform PCM and deliver to roles. Returns cache results.

        Encoding is parallelized across unique TransformKeys via a thread pool.
        """
        # Collect unique encoding tasks: tkey -> (transformer, pcm_data, output_ts, duration_us)
        encode_tasks: dict[TransformKey, tuple[AudioTransformer | None, bytes, int, int]] = {}

        for pcm_key, roles_list in roles_by_pcm.items():
            channel_id, rate, depth, channels = pcm_key
            pcm_data, output_ts = resampled_pcm[pcm_key]

            # Calculate duration for passthrough (no transformer)
            frame_stride = (depth // 8) * channels
            sample_count = len(pcm_data) // frame_stride if frame_stride > 0 else 0
            duration_us = int(sample_count * 1_000_000 / rate) if rate > 0 else 0

            grouped_by_key: defaultdict[
                TransformKey, list[tuple[SendspinClient, Role, AudioRequirements]]
            ] = defaultdict(list)
            for client, role, req in roles_list:
                tkey = self._build_transform_key(req, channel_id, role)
                grouped_by_key[tkey].append((client, role, req))

            for tkey, grouped in grouped_by_key.items():
                if self._catchup_state.get(tkey) == "catching_up":
                    continue
                if tkey in encode_tasks:
                    continue
                transformer = grouped[0][2].transformer
                encode_tasks[tkey] = (transformer, pcm_data, output_ts, duration_us)

        # TransformKey -> list of (data, timestamp_us, duration_us)
        transformed: dict[TransformKey, list[tuple[bytes, int, int]]] = {}

        if encode_tasks:
            # Multiple encoders: run in parallel via thread pool
            executor = _get_encode_executor()
            # Use the real running loop for run_in_executor (self._loop may be mocked)
            running_loop = asyncio.get_running_loop()

            async def run_encode(
                tkey: TransformKey,
                transformer: AudioTransformer | None,
                pcm_data: bytes,
                output_ts: int,
                duration_us: int,
            ) -> tuple[TransformKey, list[tuple[bytes, int, int]]]:
                result = await running_loop.run_in_executor(
                    executor,
                    _encode_for_transform_key,
                    transformer,
                    pcm_data,
                    output_ts,
                    duration_us,
                )
                return (tkey, result)

            tasks = [
                run_encode(tkey, transformer, pcm_data, output_ts, duration_us)
                for tkey, (transformer, pcm_data, output_ts, duration_us) in encode_tasks.items()
            ]
            results = await asyncio.gather(*tasks)
            transformed = dict(results)

        cache_results: defaultdict[TransformKey, list[CachedChunk]] = defaultdict(list)

        for tkey, frame_list in transformed.items():
            for data, ts, dur in frame_list:
                cached = CachedChunk(
                    timestamp_us=ts, duration_us=dur, payload=data, byte_count=len(data)
                )
                cache_results[tkey].append(cached)

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
                if chunk.timestamp_us >= now_us:
                    break
                prune_count += 1

            if prune_count:
                self._role_chunk_cache[key] = chunks[prune_count:]
                self._adjust_role_chunk_cursors(key, prune_count)

            if not self._role_chunk_cache[key]:
                del self._role_chunk_cache[key]
                self._drop_role_chunk_cursors(key)

        for channel_int in list(self._pcm_chunk_cache.keys()):
            pcm_chunks = self._pcm_chunk_cache[channel_int]
            while pcm_chunks and pcm_chunks[0].timestamp_us < now_us:
                pcm_chunks.popleft()
            if not pcm_chunks:
                self._pcm_chunk_cache.pop(channel_int, None)

    def _adjust_role_chunk_cursors(self, tkey: TransformKey, removed: int) -> None:
        if removed <= 0:
            return
        for role, cursors in list(self._role_chunk_cursors.items()):
            if tkey not in cursors:
                continue
            cursors[tkey] = max(cursors[tkey] - removed, 0)
            if cursors[tkey] == 0 and not self._role_chunk_cache.get(tkey):
                cursors.pop(tkey, None)
            if not cursors:
                self._role_chunk_cursors.pop(role, None)

    def _drop_role_chunk_cursors(self, tkey: TransformKey) -> None:
        for role, cursors in list(self._role_chunk_cursors.items()):
            if tkey in cursors:
                cursors.pop(tkey, None)
            if not cursors:
                self._role_chunk_cursors.pop(role, None)

    def _get_roles_by_transform(
        self,
    ) -> defaultdict[TransformKey, list[tuple[SendspinClient, Role, AudioRequirements]]]:
        roles_by_transform: defaultdict[
            TransformKey, list[tuple[SendspinClient, Role, AudioRequirements]]
        ] = defaultdict(list)
        for client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req is None:
                continue
            channel_id = req.channel_id or MAIN_CHANNEL
            tkey = self._build_transform_key(req, channel_id, role)
            roles_by_transform[tkey].append((client, role, req))
        return roles_by_transform

    def _ensure_role_started(self, role: Role) -> None:
        if role in self._started_roles:
            return
        role.on_stream_start()
        self._started_roles.add(role)

    def _pump_role_cache(
        self,
        role: Role,
        tkey: TransformKey,
        cached: list[CachedChunk],
        now_us: int,
    ) -> bool:
        """Pump cached chunks to a role, up to _PUMP_BATCH_SIZE at a time.

        Returns True if more work remains (batch yielding), False if done.
        """
        role_cursors = self._role_chunk_cursors.setdefault(role, {})
        cursor = role_cursors.get(tkey, 0)
        skipped_late = 0
        chunks_sent = 0

        while cursor < len(cached):
            cached_chunk = cached[cursor]
            if cached_chunk.timestamp_us < now_us:
                skipped_late += 1
                cursor += 1
                continue

            self._ensure_role_started(role)

            chunk = AudioChunk(
                data=cached_chunk.payload,
                timestamp_us=cached_chunk.timestamp_us,
                duration_us=cached_chunk.duration_us,
                byte_count=cached_chunk.byte_count,
            )
            role.on_audio_chunk(chunk)
            cursor += 1
            chunks_sent += 1

            # Yield to event loop after batch to prevent stalling
            if chunks_sent >= _PUMP_BATCH_SIZE:
                break

        if skipped_late > 0:
            _LOGGER.debug(
                "Pump skipped %s late chunk(s) for role %s (ts < now_us=%s)",
                skipped_late,
                role.role_family,
                now_us,
            )

        role_cursors[tkey] = cursor
        return cursor < len(cached)

    def _pump_cached_chunks(self) -> None:
        """Pump cached chunks to all roles, yielding to event loop between batches."""
        if self._is_stopped:
            return
        self._cache_pump_handle = None

        now_us = self._clock.now_us()
        roles_by_transform = self._get_roles_by_transform()
        has_more_work = False

        for tkey, roles in roles_by_transform.items():
            cached = self._role_chunk_cache.get(tkey)
            if not cached:
                continue
            for _client, role, _req in roles:
                more = self._pump_role_cache(role, tkey, cached, now_us)
                if more:
                    has_more_work = True

        # Schedule immediate continuation if more work remains (batch yielding)
        if has_more_work and not self._is_stopped:
            self._cache_pump_handle = self._loop.call_soon(self._pump_cached_chunks)

    def _trigger_cache_pump(self) -> None:
        if self._cache_pump_handle is not None:
            self._cache_pump_handle.cancel()
            self._cache_pump_handle = None
        self._pump_cached_chunks()

    def on_role_leave(self, role: Role) -> None:
        """Remove role-specific state so re-joins get fresh stream/start."""
        self._started_roles.discard(role)
        self._role_chunk_cursors.pop(role, None)
        req = role.get_audio_requirements()
        if req is not None:
            channel_id = req.channel_id or MAIN_CHANNEL
            tkey = self._build_transform_key(req, channel_id, role)
            if not self._other_roles_use_transform_key(tkey, role):
                self._role_chunk_cache.pop(tkey, None)
                self._drop_role_chunk_cursors(tkey)
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
        for cache_key in list(self._transform_key_cache.keys()):
            if cache_key[0] == role_id:
                self._transform_key_cache.pop(cache_key, None)

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
        delay_s = role.get_join_delay_s()
        if delay_s > 0:
            self._loop.call_later(delay_s, self._do_role_join, role)
            return
        self._do_role_join(role)

    def _do_role_join(self, role: Role) -> None:
        """Execute role join with cached chunk replay."""
        if self._is_stopped:
            return
        req = role.get_audio_requirements()
        if req is None:
            return

        channel_id = req.channel_id or MAIN_CHANNEL

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
                self._catchup_state[cache_key] = "catching_up"
                self._catchup_roles[cache_key] = {role}
                self._catchup_tasks[cache_key] = create_task(
                    self._start_catchup_encoding(role, req, channel_id, cache_key)
                )
                return

            if self._channel_timing:
                self._ensure_role_started(role)
            return

        now_us = self._clock.now_us()
        min_timestamp_us = now_us + LATE_JOINER_MIN_LEAD_US

        start_index = 0
        for chunk in cached:
            if chunk.timestamp_us >= min_timestamp_us:
                break
            start_index += 1

        if start_index >= len(cached):
            if self._channel_timing:
                self._ensure_role_started(role)
            return

        role_cursors = self._role_chunk_cursors.setdefault(role, {})
        if role_cursors.get(cache_key, 0) < start_index:
            role_cursors[cache_key] = start_index

        if _LOGGER.isEnabledFor(logging.DEBUG):
            first_ts = cached[start_index].timestamp_us
            last_ts = cached[-1].timestamp_us
            _LOGGER.debug(
                "Late join catch-up via role hook: chunks=%s ts_range=%s..%s",
                len(cached) - start_index,
                first_ts,
                last_ts,
            )

        if self._channel_timing:
            self._ensure_role_started(role)
        has_more = self._pump_role_cache(role, cache_key, cached, now_us)
        if has_more and self._cache_pump_handle is None:
            self._cache_pump_handle = self._loop.call_soon(self._pump_cached_chunks)

    def _other_roles_use_transform_key(self, cache_key: TransformKey, exclude_role: Role) -> bool:
        """Check if any other connected role uses the same TransformKey."""
        for client in self._group.clients:
            if not client.is_connected:
                continue
            for role in client.active_roles:
                if role is exclude_role:
                    continue
                req = role.get_audio_requirements()
                if req is None:
                    continue
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                if tkey == cache_key:
                    return True
        return False

    def _encode_pcm_sequence(
        self,
        pcm_chunks: list[CachedPCMChunk],
        encoder: AudioTransformer | None,
        req: AudioRequirements,
        channel_id: UUID,
    ) -> list[CachedChunk]:
        """Resample PCM chunks to the target format and encode them sequentially."""
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

            resampled_pcm, output_ts = _resample_pcm_standalone(
                resampler_state,
                chunk.pcm_data,
                source_format,
                chunk.timestamp_us,
            )
            if not resampled_pcm:
                continue

            frame_stride = (req.bit_depth // 8) * req.channels
            sample_count = len(resampled_pcm) // frame_stride if frame_stride > 0 else 0
            duration_us = int(sample_count * 1_000_000 / req.sample_rate) if sample_count > 0 else 0

            encoded_frames = _encode_for_transform_key(
                encoder,
                resampled_pcm,
                output_ts,
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
            pcm_chunks = list(self._pcm_chunk_cache.get(channel_int, []))
            now_us = self._clock.now_us()
            min_ts = now_us + LATE_JOINER_MIN_LEAD_US
            eligible = [chunk for chunk in pcm_chunks if chunk.timestamp_us >= min_ts]

            if not eligible:
                if self._channel_timing:
                    for r in self._catchup_roles.get(cache_key, {role}):
                        self._ensure_role_started(r)
                return

            encoded = await self._loop.run_in_executor(
                _get_encode_executor(),
                self._encode_pcm_sequence,
                eligible,
                encoder,
                req,
                channel_id,
            )

            if encoded:
                self._role_chunk_cache[cache_key].extend(encoded)

            if encoded:
                last_encoded_end_us = encoded[-1].timestamp_us + encoded[-1].duration_us
            else:
                last_encoded_end_us = eligible[-1].timestamp_us + eligible[-1].duration_us

            while True:
                await asyncio.sleep(0)

                now_us = self._clock.now_us()
                target_ts = now_us + LATE_JOINER_MIN_LEAD_US
                if last_encoded_end_us >= target_ts:
                    break

                new_pcm = [
                    chunk
                    for chunk in self._pcm_chunk_cache.get(channel_int, [])
                    if chunk.timestamp_us >= last_encoded_end_us
                ]
                if not new_pcm:
                    await asyncio.sleep(0.01)
                    continue

                new_encoded = await self._loop.run_in_executor(
                    _get_encode_executor(),
                    self._encode_pcm_sequence,
                    new_pcm,
                    encoder,
                    req,
                    channel_id,
                )

                if new_encoded:
                    self._role_chunk_cache[cache_key].extend(new_encoded)
                    last_encoded_end_us = new_encoded[-1].timestamp_us + new_encoded[-1].duration_us

            for r in self._catchup_roles.get(cache_key, {role}):
                self._role_chunk_cursors.setdefault(r, {})[cache_key] = 0
                self._ensure_role_started(r)

            self._catchup_state[cache_key] = "live"
            self._trigger_cache_pump()
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
        Stop the stream.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        Flushes remaining audio from transformers, then sends stream/end message
        to all roles via hooks.
        """
        if self._is_stopped:
            return
        self._is_stopped = True
        if self._cache_pump_handle is not None:
            self._cache_pump_handle.cancel()
            self._cache_pump_handle = None

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
        self._role_chunk_cursors.clear()
        self._cancel_catchup_tasks()
        self._pcm_chunk_cache.clear()

    def clear(self) -> None:
        """
        Clear all pending audio and reset timing.

        This is used for seek operations where buffered audio is discarded.
        Sends stream/clear to all roles via hooks.
        """
        # Clear pending audio
        self._channel_buffers.clear()

        # Reset per-channel timing
        self._channel_timing.clear()

        # Clear chunk cache
        self._role_chunk_cache.clear()
        self._role_chunk_cursors.clear()
        self._pcm_chunk_cache.clear()
        self._cancel_catchup_tasks()
        if self._cache_pump_handle is not None:
            self._cache_pump_handle.cancel()
            self._cache_pump_handle = None

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
