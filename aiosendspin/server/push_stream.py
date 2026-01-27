"""Push-based audio streaming API."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from aiosendspin.server.audio import AudioFormat, _get_av, _resolve_audio_format
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles import AudioChunk

if TYPE_CHECKING:
    import av

    from aiosendspin.server.channels import ChannelRouter
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.clock import Clock
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles import AudioRequirements, Role

_LOGGER = logging.getLogger(__name__)

# Default initial delay before first audio plays (microseconds)
DEFAULT_INITIAL_DELAY_US = 250_000  # 250ms


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


class StreamStoppedError(Exception):
    """Raised when trying to commit audio on a stopped stream."""


class PushStream:
    """
    Push-based audio streaming API.

    This class provides a push-based interface for streaming audio to players.
    Audio is prepared via prepare_audio(), then committed and sent via commit_audio().
    Backpressure is handled via wait_for_buffer_space() and timeline shifting.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        clock: Clock,
        group: SendspinGroup,
        channel_router: ChannelRouter,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            clock: Time source used for timestamping.
            group: Group this stream belongs to.
            channel_router: Router for channel assignments.
        """
        self._loop = loop
        self._clock = clock
        self._group = group
        self._channel_router = channel_router
        self._is_stopped = False
        # Pending audio per channel: channel_id -> (pcm_bytes, audio_format)
        self._channel_buffers: dict[UUID, tuple[bytes, AudioFormat]] = {}
        # Per-channel timing: channel_id -> next_chunk_start_us
        self._channel_timing: dict[UUID, int] = {}
        # Role-based streaming tracking (for hook-based flow)
        self._started_roles: set[Role] = set()
        self._backpressured_roles: set[Role] = set()
        # Inline resamplers for role-based audio delivery
        self._resamplers: dict[_ResamplerKey, _ResamplerState] = {}
        # Role-based chunk cache: (channel_id, transformer_id) -> list of cached chunks
        self._role_chunk_cache: dict[tuple[UUID, int], list[CachedChunk]] = {}

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

    def _get_or_create_resampler(
        self,
        key: _ResamplerKey,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> _ResamplerState:
        """Get existing resampler or create a new one."""
        if key in self._resamplers:
            return self._resamplers[key]

        av = _get_av()

        # Get source format params
        _source_bytes_per_sample, source_av_format, source_layout = _resolve_audio_format(
            source_format
        )

        # Get target format params
        target_bytes_per_sample, target_av_format, target_layout = _resolve_audio_format(
            target_format
        )

        # Create resampler
        resampler = av.AudioResampler(
            format=target_av_format,
            layout=target_layout,
            rate=target_format.sample_rate,
        )

        state = _ResamplerState(
            key=key,
            resampler=resampler,
            source_av_format=source_av_format,
            source_av_layout=source_layout,
            target_av_format=target_av_format,
            target_layout=target_layout,
            target_frame_stride=target_bytes_per_sample * target_format.channels,
        )
        self._resamplers[key] = state
        return state

    def _resample_pcm(
        self,
        resampler_state: _ResamplerState,
        pcm_data: bytes,
        source_format: AudioFormat,
        input_timestamp_us: int,
    ) -> tuple[bytes, int]:
        """Resample PCM data to the target format.

        Args:
            resampler_state: The resampler state to use.
            pcm_data: Source PCM bytes.
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
        sample_count = len(pcm_data) // frame_stride

        if sample_count == 0:
            return b"", resampler_state.pending_timestamp_us

        # Create input frame
        frame = av.AudioFrame(
            format=resampler_state.source_av_format,
            layout=resampler_state.source_av_layout,
            samples=sample_count,
        )
        frame.sample_rate = source_format.sample_rate
        frame.planes[0].update(pcm_data)

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

    def has_pending_audio(self) -> bool:
        """Return True if there is pending audio to commit."""
        return len(self._channel_buffers) > 0

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

    async def commit_audio(self) -> int:
        """
        Encode and send all prepared audio to players.

        This is an asynchronous method that:
        1. Encodes prepared PCM for each required format
        2. Applies backpressure (timeline shift if needed)
        3. Assigns timestamps to encoded chunks
        4. Sends chunks to connected players via role hooks

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

        # Initialize timing for new channels
        now_us = self._clock.now_us()
        for channel_id in prepared:
            if channel_id not in self._channel_timing:
                self._channel_timing[channel_id] = now_us + DEFAULT_INITIAL_DELAY_US

        # Calculate approximate byte count for backpressure (use total of all channels)
        total_bytes = sum(len(pcm) for pcm, _ in prepared.values())

        # Apply backpressure: query roles with buffer tracking
        max_wait_us = self._calculate_backpressure(total_bytes)
        if max_wait_us > 0:
            # Shift all channel timings equally for group synchronization
            for channel_id in self._channel_timing:
                self._channel_timing[channel_id] += max_wait_us

        # If audio production stalls (e.g., the upstream source blocks), the scheduled
        # play timeline can drift into the past. Rebase the timeline so new audio is
        # always scheduled with at least the default initial delay from "now".
        min_timing_us = min(self._channel_timing.values())
        target_min_us = now_us + DEFAULT_INITIAL_DELAY_US
        if min_timing_us < target_min_us:
            shift_us = target_min_us - min_timing_us
            for channel_id in self._channel_timing:
                self._channel_timing[channel_id] += shift_us

        # Capture play_start_us for each channel before advancing
        channel_play_start: dict[UUID, int] = {}
        for channel_id in prepared:
            channel_play_start[channel_id] = self._channel_timing[channel_id]

        # Advance each channel's timing by its duration
        for channel_id, duration_us in durations_us.items():
            self._channel_timing[channel_id] += duration_us

        # Role-based audio delivery via hooks
        role_cache_results = self._deliver_audio_to_roles(prepared, channel_play_start)
        # Merge role-based cache results into the cache
        for cache_key, chunks in role_cache_results.items():
            self._role_chunk_cache.setdefault(cache_key, []).extend(chunks)

        # Prune old chunks from cache
        self._prune_role_chunk_cache()

        # Return earliest play_start_us
        return min(channel_play_start.values())

    def _calculate_backpressure(self, byte_count: int) -> int:
        """
        Calculate backpressure delay based on client buffer capacity.

        Args:
            byte_count: Approximate bytes being sent to players.

        Returns:
            Maximum wait time in microseconds across all clients.
        """
        max_wait_us = 0
        for client, _role in self._get_audio_roles():
            if client.buffer_tracker is not None:
                wait_us = client.buffer_tracker.time_until_capacity(byte_count)
                if wait_us > 0 and _LOGGER.isEnabledFor(logging.DEBUG):
                    _LOGGER.debug(
                        "Backpressure from %s: wait_us=%s buffered=%s capacity=%s",
                        client.client_id,
                        wait_us,
                        client.buffer_tracker.buffered_bytes,
                        client.buffer_tracker.capacity_bytes,
                    )
                max_wait_us = max(max_wait_us, wait_us)
        return max_wait_us

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
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ] = {}

        for client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req is None:
                continue

            channel_id = req.channel_id or MAIN_CHANNEL
            if channel_id not in prepared:
                continue

            pcm_key = (channel_id, req.sample_rate, req.bit_depth, req.channels)
            roles_by_pcm.setdefault(pcm_key, []).append((client, role, req))

        return roles_by_pcm

    def _resample_for_roles(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
    ) -> dict[tuple[UUID, int, int, int], tuple[bytes, int]]:
        """Resample PCM once per unique PCM key. Returns (channel, rate, depth, ch) -> (pcm, ts)."""
        resampled: dict[tuple[UUID, int, int, int], tuple[bytes, int]] = {}

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

            resampler_state = self._get_or_create_resampler(
                resampler_key, source_format, target_format
            )
            pcm_out, output_ts = self._resample_pcm(
                resampler_state, source_pcm, source_format, input_timestamp_us
            )
            resampled[pcm_key] = (pcm_out, output_ts)

        return resampled

    def _transform_and_deliver(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        resampled_pcm: dict[tuple[UUID, int, int, int], tuple[bytes, int]],
    ) -> dict[tuple[UUID, int], list[CachedChunk]]:
        """Transform PCM and deliver to roles. Returns cache results."""
        # (channel_id, transformer_id) -> list of (data, timestamp_us, duration_us)
        transformed: dict[tuple[UUID, int], list[tuple[bytes, int, int]]] = {}
        # Track roles per transform key
        roles_by_transform: dict[
            tuple[UUID, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ] = {}

        for pcm_key, roles_list in roles_by_pcm.items():
            channel_id, rate, depth, channels = pcm_key
            pcm_data, output_ts = resampled_pcm[pcm_key]

            # Calculate duration for passthrough (no transformer)
            frame_stride = (depth // 8) * channels
            sample_count = len(pcm_data) // frame_stride if frame_stride > 0 else 0
            duration_us = int(sample_count * 1_000_000 / rate) if rate > 0 else 0

            # Group by transformer
            by_transformer: dict[int, list[tuple[SendspinClient, Role, AudioRequirements]]] = {}
            for client, role, req in roles_list:
                tid = id(req.transformer) if req.transformer else 0
                by_transformer.setdefault(tid, []).append((client, role, req))

            for tid, grouped in by_transformer.items():
                tkey = (channel_id, tid)
                roles_by_transform.setdefault(tkey, []).extend(grouped)

                if tkey not in transformed:
                    transformer = grouped[0][2].transformer
                    if transformer is None:
                        # No transformer - passthrough as single frame
                        transformed[tkey] = [(pcm_data, output_ts, duration_us)]
                    else:
                        # Transformer returns list[bytes] - one tuple per frame
                        frames = transformer.process(pcm_data, output_ts, duration_us)

                        # Get base timestamp AFTER processing. Transformers track output
                        # timeline via pending_timestamp_us. Getting it after process()
                        # ensures gap detection (which resets the timeline) is applied.
                        # pending_timestamp_us points to the NEXT frame's timestamp,
                        # so base_ts = pending - (num_frames * frame_dur).
                        frame_list: list[tuple[bytes, int, int]] = []
                        if frames:
                            frame_dur = transformer.frame_duration_us
                            base_ts = output_ts
                            if hasattr(transformer, "pending_timestamp_us"):
                                pending = transformer.pending_timestamp_us
                                if pending is not None:
                                    base_ts = pending - (len(frames) * frame_dur)
                            for i, frame_data in enumerate(frames):
                                frame_ts = base_ts + (i * frame_dur)
                                frame_list.append((frame_data, frame_ts, frame_dur))
                        transformed[tkey] = frame_list

        # Deliver and cache
        cache_results: dict[tuple[UUID, int], list[CachedChunk]] = {}

        for tkey, frame_list in transformed.items():
            for data, ts, dur in frame_list:
                chunk = AudioChunk(
                    data=data, timestamp_us=ts, duration_us=dur, byte_count=len(data)
                )
                cached = CachedChunk(
                    timestamp_us=ts,
                    duration_us=dur,
                    payload=data,
                    byte_count=len(data),
                )
                cache_results.setdefault(tkey, []).append(cached)

                for _client, role, _req in roles_by_transform.get(tkey, []):
                    if role not in self._started_roles:
                        role.on_stream_start()
                        self._started_roles.add(role)

                    if not role.on_audio_chunk(chunk):
                        self._backpressured_roles.add(role)

        return cache_results

    def _deliver_audio_to_roles(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
    ) -> dict[tuple[UUID, int], list[CachedChunk]]:
        """
        Deliver audio to roles using the hook-based flow.

        This method:
        1. Groups roles by unique PCM requirements
        2. Resamples source PCM to each unique target format
        3. Transforms and delivers via role.on_audio_chunk()

        Returns:
            Dict of (channel_id, transformer_id) -> list of CachedChunk for late joiners.
        """
        roles_by_pcm = self._group_roles_by_pcm_requirements(prepared)
        if not roles_by_pcm:
            return {}

        resampled = self._resample_for_roles(roles_by_pcm, prepared, channel_play_start)
        return self._transform_and_deliver(roles_by_pcm, resampled)

    def _prune_role_chunk_cache(self) -> None:
        """Remove old chunks from the role-based cache."""
        now_us = self._clock.now_us()

        for key in list(self._role_chunk_cache.keys()):
            # Filter out chunks older than now
            self._role_chunk_cache[key] = [
                chunk for chunk in self._role_chunk_cache[key] if chunk.timestamp_us >= now_us
            ]
            # Remove empty lists
            if not self._role_chunk_cache[key]:
                del self._role_chunk_cache[key]

    async def wait_for_buffer_space(self) -> None:
        """
        Wait until there is buffer space available on clients.

        This is useful for throttling audio production to match
        player consumption rates. Uses an estimated chunk size
        to determine buffer capacity needs.
        """
        # Estimate chunk size: 25ms of 48kHz stereo 16-bit PCM = 4800 bytes
        # This is a reasonable default for typical audio streaming
        estimated_chunk_bytes = 4800

        max_wait_us = 0
        for client, _role in self._get_audio_roles():
            if client.buffer_tracker is not None:
                wait_us = client.buffer_tracker.time_until_capacity(estimated_chunk_bytes)
                max_wait_us = max(max_wait_us, wait_us)

        if max_wait_us > 0:
            # Convert microseconds to seconds for asyncio.sleep
            await asyncio.sleep(max_wait_us / 1_000_000)

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
        req = role.get_audio_requirements()
        if req is None:
            return

        transformer = req.transformer
        channel_id = req.channel_id or MAIN_CHANNEL

        # Get cached chunks for this transformer from the role-based cache
        transformer_id = id(transformer) if transformer else 0
        cache_key = (channel_id, transformer_id)
        cached = self._role_chunk_cache.get(cache_key, [])

        if not cached:
            return

        # Filter to chunks in the future (late joiner timing)
        now_us = self._clock.now_us()
        min_timestamp_us = now_us + LATE_JOINER_MIN_LEAD_US
        future_chunks = [c for c in cached if c.timestamp_us >= min_timestamp_us]

        if not future_chunks:
            return

        if _LOGGER.isEnabledFor(logging.DEBUG):
            first_ts = future_chunks[0].timestamp_us
            last_ts = future_chunks[-1].timestamp_us
            _LOGGER.debug(
                "Late join catch-up via role hook: chunks=%s ts_range=%s..%s",
                len(future_chunks),
                first_ts,
                last_ts,
            )

        # Send stream/start via hook
        role.on_stream_start()
        self._started_roles.add(role)

        # Send cached chunks via hooks
        for cached_chunk in future_chunks:
            chunk = AudioChunk(
                data=cached_chunk.payload,
                timestamp_us=cached_chunk.timestamp_us,
                duration_us=cached_chunk.duration_us,
                byte_count=cached_chunk.byte_count,
            )
            if not role.on_audio_chunk(chunk):
                # Backpressure - stop sending
                self._backpressured_roles.add(role)
                break

    def stop(self) -> None:
        """
        Stop the stream.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        Flushes remaining audio from transformers, then sends stream/end message
        to all roles via hooks.
        """
        self._is_stopped = True

        # Flush remaining audio from transformers and reset them
        flushed_transformers: set[int] = set()
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req and req.transformer:
                tid = id(req.transformer)
                if tid not in flushed_transformers:
                    flushed_transformers.add(tid)
                    final_frames = req.transformer.flush()
                    if final_frames:
                        frame_duration_us = req.transformer.frame_duration_us
                        for frame_data in final_frames:
                            chunk = AudioChunk(
                                data=frame_data,
                                timestamp_us=0,  # Timestamp doesn't matter at stream end
                                duration_us=frame_duration_us,
                                byte_count=len(frame_data),
                            )
                            role.on_audio_chunk(chunk)
                    # Reset transformer for next stream
                    req.transformer.reset()

        # Send stream/end to all roles with audio requirements via hooks
        for _client, role in self._get_audio_roles():
            role.on_stream_end()

        # Clear role tracking state
        self._started_roles.clear()
        self._backpressured_roles.clear()

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

        # Reset inline resamplers
        self._resamplers.clear()

        # Reset transformers so they don't carry stale timestamp state
        reset_transformers: set[int] = set()
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req and req.transformer:
                tid = id(req.transformer)
                if tid not in reset_transformers:
                    reset_transformers.add(tid)
                    req.transformer.reset()

        # Clear role tracking state
        self._started_roles.clear()
        self._backpressured_roles.clear()

        # Send stream/clear to all roles with audio requirements via hooks
        for _client, role in self._get_audio_roles():
            role.on_stream_clear()
