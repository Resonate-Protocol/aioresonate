"""Audio playback for the Resonate CLI."""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import sounddevice
from sounddevice import CallbackFlags

from aioresonate.client import PCMFormat

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _QueuedChunk:
    """Represents a queued audio chunk with timing information."""

    server_timestamp_us: int
    """Server timestamp when this chunk should start playing."""
    audio_data: bytes
    """Raw PCM audio bytes."""


class AudioPlayer:
    """
    Audio player for the Resonate CLI with time synchronization support.

    This player accepts audio chunks with server timestamps and dynamically
    computes playback times using a time synchronization function. This allows
    for accurate synchronization even when the time base changes during playback.

    Attributes:
        _loop: The asyncio event loop used for scheduling.
        _compute_client_time: Function that converts server timestamps to client
            timestamps (monotonic loop time), accounting for clock drift, offset,
            and static delay.
        _compute_server_time: Function that converts client timestamps (monotonic
            loop time) to server timestamps (inverse of _compute_client_time).
    """

    _loop: asyncio.AbstractEventLoop
    _compute_client_time: Callable[[int], int]
    _compute_server_time: Callable[[int], int]

    _MIN_CHUNKS_TO_START: Final[int] = 16
    """Minimum chunks buffered before starting playback to absorb network jitter."""
    _MIN_CHUNKS_TO_MAINTAIN: Final[int] = 8
    """Minimum chunks to maintain during playback to avoid underruns."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        compute_client_time: Callable[[int], int],
        compute_server_time: Callable[[int], int],
    ) -> None:
        """
        Initialize the audio player.

        Args:
            loop: The asyncio event loop to use for scheduling.
            compute_client_time: Function that converts server timestamps to client
                timestamps (monotonic loop time), accounting for clock drift, offset,
                and static delay.
            compute_server_time: Function that converts client timestamps (monotonic
                loop time) to server timestamps (inverse of compute_client_time).
        """
        self._loop = loop
        self._compute_client_time = compute_client_time
        self._compute_server_time = compute_server_time
        self._format: PCMFormat | None = None
        self._queue: asyncio.Queue[_QueuedChunk] = asyncio.Queue()
        self._stream: sounddevice.RawOutputStream | None = None
        self._closed = False
        self._stream_started = False
        self._first_real_chunk = True  # Flag to initialize timing from first chunk

        # Partial chunk tracking (to avoid discarding partial chunks)
        self._current_chunk: _QueuedChunk | None = None
        self._current_chunk_offset = 0

        # Track expected next chunk timestamp for intelligent gap/overlap handling
        self._expected_next_timestamp: int | None = None

        # Underrun tracking
        self._underrun_count = 0
        self._last_buffer_warning_us = 0

        # Track queued audio duration instead of just item count
        self._queued_duration_us = 0

        # DAC timing for accurate playback position tracking
        self._dac_loop_calibrations: collections.deque[tuple[int, int]] = collections.deque(
            maxlen=100
        )
        # Recent [(dac_time_us, loop_time_us), ...] pairs for DAC↔Loop mapping
        self._last_known_playback_position_us: int = 0
        # Current playback position in server timestamp space
        self._last_dac_calibration_time_us: int = 0
        # Last loop time when we calibrated DAC↔Loop mapping

        # Scheduled start anchoring
        self._scheduled_start_loop_time_us: int | None = None
        self._scheduled_start_dac_time_us: int | None = None
        self._waiting_for_start: bool = True

        # Server timeline cursor for the next input frame to be consumed
        self._server_ts_cursor_us: int = 0
        self._server_ts_cursor_remainder: int = 0  # fractional accumulator for microseconds

        # First-chunk and re-anchor tracking
        self._first_server_timestamp_us: int | None = None
        self._early_start_suspect: bool = False
        self._has_reanchored: bool = False

        # Low-overhead drift/phase correction scheduling (sample drop/insert)
        self._insert_every_n_frames: int = 0
        self._drop_every_n_frames: int = 0
        self._frames_until_next_insert: int = 0
        self._frames_until_next_drop: int = 0
        self._last_output_frame: bytes = b""

        # Phase error smoothing and re-anchor cooldown
        self._phase_error_ema_us: float = 0.0
        self._phase_error_ema_init: bool = False
        self._last_reanchor_loop_time_us: int = 0
        self._last_phase_error_log_us: int = 0  # Rate limit phase error logging

    def set_format(self, pcm_format: PCMFormat) -> None:
        """Configure the audio output format."""
        self._format = pcm_format
        self._close_stream()

        # Reset state on format change
        self._stream_started = False
        self._first_real_chunk = True

        dtype = "int16"
        # Use low latency for accurate playback timing
        # With early chunk arrival (5+ seconds), we can use aggressive low-latency settings
        blocksize = 2048  # ~46ms at 44.1kHz - balance between latency and stability
        self._stream = sounddevice.RawOutputStream(
            samplerate=pcm_format.sample_rate,
            channels=pcm_format.channels,
            dtype=dtype,
            blocksize=blocksize,
            callback=self._audio_callback,
            latency="low",  # Low latency for minimal device-induced delay
        )
        logger.info("Audio stream configured: blocksize=%d, latency=low", blocksize)

    async def stop(self) -> None:
        """Stop playback and release resources."""
        self._closed = True
        self._close_stream()

    def clear(self) -> None:
        """Drop all queued audio chunks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Reset playback state
        self._first_real_chunk = True
        self._current_chunk = None
        self._current_chunk_offset = 0
        self._expected_next_timestamp = None
        self._underrun_count = 0
        self._queued_duration_us = 0
        # Reset timing calibration for fresh start
        self._dac_loop_calibrations.clear()
        self._last_known_playback_position_us = 0
        self._last_dac_calibration_time_us = 0
        self._scheduled_start_loop_time_us = None
        self._scheduled_start_dac_time_us = None
        self._waiting_for_start = True
        self._server_ts_cursor_us = 0
        self._server_ts_cursor_remainder = 0
        self._first_server_timestamp_us = None
        self._early_start_suspect = False
        self._has_reanchored = False
        self._insert_every_n_frames = 0
        self._drop_every_n_frames = 0
        self._frames_until_next_insert = 0
        self._frames_until_next_drop = 0
        self._last_output_frame = b""
        self._phase_error_ema_us = 0.0
        self._phase_error_ema_init = False
        self._last_reanchor_loop_time_us = 0
        self._last_phase_error_log_us = 0

    def _audio_callback(
        self,
        outdata: memoryview,
        frames: int,
        time: Any,
        status: CallbackFlags,
    ) -> None:
        """
        Audio callback invoked by sounddevice when output buffer needs filling.

        Args:
            outdata: Output buffer to fill with audio data.
            frames: Number of frames requested.
            time: CFFI cdata structure with timing info (outputBufferDacTime, etc).
            status: Status flags (underrun, overflow, etc.).
        """
        if status:
            # Detect underflow and immediately re-anchor to avoid long desync
            if status.input_underflow or status.output_underflow:
                logger.warning("Audio underflow detected; re-anchoring playback")
                self.clear()
            else:
                logger.debug("Audio callback status: %s", status)

        assert self._format is not None

        # Capture exact DAC output time and update playback position
        self._update_playback_position_from_dac(time)

        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")
        bytes_written = 0

        try:
            # Pre-start gating: fill silence until scheduled start time in DAC domain
            if self._waiting_for_start:
                if self._scheduled_start_dac_time_us is not None:
                    try:
                        dac_now_us = int(time.outputBufferDacTime * 1_000_000)
                    except (AttributeError, TypeError):
                        dac_now_us = 0

                    if dac_now_us > 0:
                        delta_us = self._scheduled_start_dac_time_us - dac_now_us
                        if delta_us > 0:
                            frames_until_start = int(
                                (delta_us * self._format.sample_rate + 999_999) // 1_000_000
                            )
                            frames_to_silence = min(frames_until_start, frames)
                            silence_bytes = frames_to_silence * self._format.frame_size
                            if silence_bytes > 0:
                                output_buffer[:silence_bytes] = b"\x00" * silence_bytes
                                bytes_written += silence_bytes
                        elif delta_us < 0:
                            # Late: fast-forward by dropping input frames, unless we suspect
                            # we anchored before sync matured; in that case, keep waiting.
                            if self._early_start_suspect and not self._has_reanchored:
                                pass
                            else:
                                frames_to_drop = int(
                                    ((-delta_us) * self._format.sample_rate + 999_999) // 1_000_000
                                )
                                self._skip_input_frames(frames_to_drop)
                                self._waiting_for_start = False
                        # If we've reached/overrun the scheduled time, arm playback
                        if dac_now_us >= self._scheduled_start_dac_time_us:
                            self._waiting_for_start = False
                elif self._scheduled_start_loop_time_us is not None:
                    loop_now_us = int(self._loop.time() * 1_000_000)
                    delta_us = self._scheduled_start_loop_time_us - loop_now_us
                    if delta_us > 0:
                        frames_until_start = int(
                            (delta_us * self._format.sample_rate + 999_999) // 1_000_000
                        )
                        frames_to_silence = min(frames_until_start, frames)
                        silence_bytes = frames_to_silence * self._format.frame_size
                        if silence_bytes > 0:
                            output_buffer[:silence_bytes] = b"\x00" * silence_bytes
                            bytes_written += silence_bytes
                    else:
                        # For loop-time gating, do not drop input when late; keep waiting
                        # until we can map to DAC time or schedule updates in submit().
                        pass

            # If we're still waiting for start after gating above, output silence only
            if self._waiting_for_start:
                if bytes_written < bytes_needed:
                    silence_bytes = bytes_needed - bytes_written
                    output_buffer[bytes_written : bytes_written + silence_bytes] = (
                        b"\x00" * silence_bytes
                    )
                    bytes_written += silence_bytes
            else:
                # Simple drift control based on buffer depth (disabled for heavy actions)
                target_buffer_us = self._compute_minimum_buffer_duration()
                current_remaining_us = 0
                if self._current_chunk is not None:
                    rem_bytes = len(self._current_chunk.audio_data) - self._current_chunk_offset
                    rem_frames = rem_bytes // self._format.frame_size
                    current_remaining_us = (rem_frames * 1_000_000) // self._format.sample_rate

                buffer_depth_us = self._queued_duration_us + current_remaining_us
                delta_us = buffer_depth_us - target_buffer_us

                # Large jump handling after start: prefer doing nothing to avoid pitch
                # Determine extra frames to consume this callback (very conservative)
                CONTROL_GAIN = 0.0  # disable pitchy drop/dup by default
                max_adjust = 0
                consume_extra = int(
                    (delta_us * self._format.sample_rate / 1_000_000.0) * CONTROL_GAIN
                )
                if consume_extra > max_adjust:
                    consume_extra = max_adjust
                elif consume_extra < -max_adjust:
                    consume_extra = -max_adjust

                in_frames_target = frames + consume_extra
                in_frames_target = max(in_frames_target, 0)

                # Occasional sample drop/insert based on scheduled cadence
                if self._frames_until_next_insert <= 0 and self._insert_every_n_frames > 0:
                    self._frames_until_next_insert = self._insert_every_n_frames
                if self._frames_until_next_drop <= 0 and self._drop_every_n_frames > 0:
                    self._frames_until_next_drop = self._drop_every_n_frames

                frame_size = self._format.frame_size
                if not self._last_output_frame:
                    self._last_output_frame = b"\x00" * frame_size

                for _ in range(frames):
                    do_drop = self._drop_every_n_frames > 0 and self._frames_until_next_drop <= 0
                    do_insert = (
                        self._insert_every_n_frames > 0 and self._frames_until_next_insert <= 0
                    )

                    # Prefer drop over insert if both trigger simultaneously
                    if do_drop:
                        # Discard one input frame to speed up
                        _ = self._read_one_input_frame()
                        self._frames_until_next_drop = self._drop_every_n_frames
                        # Output last frame without reading a new one
                        frame_bytes = self._last_output_frame
                    elif do_insert:
                        # Duplicate last output frame to slow down
                        frame_bytes = self._last_output_frame
                        self._frames_until_next_insert = self._insert_every_n_frames
                    else:
                        # Normal read
                        frame_bytes = self._read_one_input_frame()
                        if frame_bytes is None:
                            frame_bytes = self._last_output_frame
                        else:
                            self._last_output_frame = frame_bytes

                    # Decrement cadence counters
                    if self._frames_until_next_insert > 0:
                        self._frames_until_next_insert -= 1
                    if self._frames_until_next_drop > 0:
                        self._frames_until_next_drop -= 1

                    # Output a single frame
                    output_buffer[bytes_written : bytes_written + frame_size] = frame_bytes
                    bytes_written += frame_size

        except Exception:
            logger.exception("Error in audio callback")
            # Fill rest with silence on error
            if bytes_written < bytes_needed:
                silence_bytes = bytes_needed - bytes_written
                output_buffer[bytes_written : bytes_written + silence_bytes] = (
                    b"\x00" * silence_bytes
                )
            # Reset partial chunk state on error
            self._current_chunk = None
            self._current_chunk_offset = 0

    def _update_playback_position_from_dac(self, time: Any) -> None:
        """Store DAC timing from audio callback.

        The actual loop time is filled in later from the async context
        in _calibrate_dac_loop_mapping() for thread safety.
        """
        try:
            dac_time_us = int(time.outputBufferDacTime * 1_000_000)
            # Store DAC time; loop time (0) will be filled in from async context
            self._dac_loop_calibrations.append((dac_time_us, 0))
            if len(self._dac_loop_calibrations) == 1:
                logger.debug("First DAC timestamp captured: %d", dac_time_us)
        except (AttributeError, TypeError):
            # time object may not have expected attributes in all backends
            logger.debug("Could not extract timing info from callback")

    def _read_one_input_frame(self) -> bytes | None:
        """Read and consume a single audio frame from the queue.

        Returns frame bytes or None if no data available.
        Updates internal cursor and buffer duration when chunks are exhausted.
        """
        if self._format is None or self._format.frame_size == 0:
            return None

        frame_size = self._format.frame_size

        # Ensure we have a current chunk
        if self._current_chunk is None:
            if self._queue.empty():
                return None
            self._current_chunk = self._queue.get_nowait()
            self._current_chunk_offset = 0
            # Initialize server cursor if needed
            if self._server_ts_cursor_us == 0:
                self._server_ts_cursor_us = self._current_chunk.server_timestamp_us

        chunk = self._current_chunk
        data = chunk.audio_data
        if self._current_chunk_offset >= len(data):
            # Should not happen, but guard
            self._advance_finished_chunk()
            return None

        start = self._current_chunk_offset
        end = start + frame_size
        end = min(end, len(data))
        frame = data[start:end]

        # Advance offsets and timeline cursor
        self._current_chunk_offset = end
        self._advance_server_cursor_frames(1)

        # If chunk finished, advance and update buffered duration tracking
        if self._current_chunk_offset >= len(data):
            self._advance_finished_chunk()

        # Ensure full frame size by padding nulls if needed (shouldn't occur normally)
        if len(frame) < frame_size:
            frame = frame + b"\x00" * (frame_size - len(frame))
        return frame

    def _advance_finished_chunk(self) -> None:
        """Called when current chunk is fully consumed to update durations/state."""
        assert self._format is not None
        if self._current_chunk is None:
            return
        data = self._current_chunk.audio_data
        chunk_frames = len(data) // self._format.frame_size
        chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
        self._queued_duration_us = max(0, self._queued_duration_us - chunk_duration_us)
        self._current_chunk = None
        self._current_chunk_offset = 0

    def _advance_server_cursor_frames(self, frames: int) -> None:
        """Advance server timeline cursor by a number of frames."""
        if self._format is None or frames <= 0:
            return
        # Accumulate microseconds precisely: add 1e6 per frame, carry by sample_rate
        self._server_ts_cursor_remainder += frames * 1_000_000
        sr = self._format.sample_rate
        if self._server_ts_cursor_remainder >= sr:
            inc_us = self._server_ts_cursor_remainder // sr
            self._server_ts_cursor_remainder = self._server_ts_cursor_remainder % sr
            self._server_ts_cursor_us += int(inc_us)

    def _skip_input_frames(self, frames_to_skip: int) -> None:
        """Discard frames from the input to reduce buffer depth quickly."""
        if self._format is None or frames_to_skip <= 0:
            return
        frame_size = self._format.frame_size
        while frames_to_skip > 0:
            if self._current_chunk is None:
                if self._queue.empty():
                    break
                self._current_chunk = self._queue.get_nowait()
                self._current_chunk_offset = 0
                if self._server_ts_cursor_us == 0:
                    self._server_ts_cursor_us = self._current_chunk.server_timestamp_us
            data = self._current_chunk.audio_data
            rem_bytes = len(data) - self._current_chunk_offset
            rem_frames = rem_bytes // frame_size
            if rem_frames <= 0:
                self._advance_finished_chunk()
                continue
            take = min(rem_frames, frames_to_skip)
            self._current_chunk_offset += take * frame_size
            self._advance_server_cursor_frames(take)
            frames_to_skip -= take
            if self._current_chunk_offset >= len(data):
                self._advance_finished_chunk()

    def _calibrate_dac_loop_mapping(self) -> None:
        """Calibrate DAC↔Loop time mapping from async context.

        Called from submit() which runs in event loop thread, allowing safe
        measurement of loop time to pair with DAC time from the callback.
        """
        if not self._dac_loop_calibrations:
            return

        # Get most recent DAC time from callback
        dac_time_us, loop_time_old_us = self._dac_loop_calibrations[-1]

        # Get current loop time (safe to call from async)
        loop_time_us = int(self._loop.time() * 1_000_000)

        # Update calibration pair
        self._dac_loop_calibrations[-1] = (dac_time_us, loop_time_us)
        self._last_dac_calibration_time_us = loop_time_us

        # Update playback position in server time using latest calibration
        try:
            # Estimate the loop time that corresponds to the captured DAC time
            loop_at_dac_us = self._estimate_loop_time_for_dac_time(dac_time_us)
            if loop_at_dac_us == 0:
                loop_at_dac_us = loop_time_us
            self._last_known_playback_position_us = self._compute_server_time(loop_at_dac_us)
        except Exception:
            pass

        # If we haven't set the DAC-anchored start yet, approximate it now
        if self._scheduled_start_dac_time_us is None and self._scheduled_start_loop_time_us:
            try:
                loop_start = self._scheduled_start_loop_time_us
                # Estimate DAC time for the scheduled loop start
                # Use last two calibrations if available for better slope estimate
                est_dac = self._estimate_dac_time_for_server_timestamp(
                    self._compute_server_time(loop_start)
                )
                if est_dac:
                    self._scheduled_start_dac_time_us = est_dac
            except Exception:
                # Fall back to mapping 1:1 when estimation fails
                self._scheduled_start_dac_time_us = self._scheduled_start_loop_time_us

    def _estimate_dac_time_for_server_timestamp(self, server_timestamp_us: int) -> int:
        """Estimate when a server timestamp will play out (in DAC time).

        Maps: server_ts → loop_time → dac_time
        """
        # Need at least one calibration point
        if self._last_dac_calibration_time_us == 0:
            return 0

        # Convert server timestamp to client loop time
        loop_time_us = self._compute_client_time(server_timestamp_us)

        # Find calibration point closest to this loop time
        if not self._dac_loop_calibrations:
            return 0

        # Use most recent calibration and previous one (if available) to estimate slope
        dac_ref_us, loop_ref_us = self._dac_loop_calibrations[-1]
        dac_prev_us, loop_prev_us = (0, 0)
        if len(self._dac_loop_calibrations) >= 2:
            dac_prev_us, loop_prev_us = self._dac_loop_calibrations[-2]

        if loop_ref_us == 0:
            # Calibration not yet filled in
            return 0

        # Estimate DAC-per-Loop slope if possible, else assume 1.0
        dac_per_loop = 1.0
        if loop_prev_us and dac_prev_us and (loop_ref_us != loop_prev_us):
            dac_per_loop = (dac_ref_us - dac_prev_us) / (loop_ref_us - loop_prev_us)
            # Clamp to sane bounds to avoid wild extrapolation
            dac_per_loop = max(0.999, min(1.001, dac_per_loop))

        return int(round(dac_ref_us + (loop_time_us - loop_ref_us) * dac_per_loop))

    def _estimate_loop_time_for_dac_time(self, dac_time_us: int) -> int:
        """Estimate loop time corresponding to a DAC time using recent calibrations."""
        if not self._dac_loop_calibrations:
            return 0
        dac_ref_us, loop_ref_us = self._dac_loop_calibrations[-1]
        if loop_ref_us == 0:
            return 0
        dac_prev_us, loop_prev_us = (0, 0)
        if len(self._dac_loop_calibrations) >= 2:
            dac_prev_us, loop_prev_us = self._dac_loop_calibrations[-2]
        loop_per_dac = 1.0
        if dac_prev_us and (dac_ref_us != dac_prev_us):
            loop_per_dac = (loop_ref_us - loop_prev_us) / (dac_ref_us - dac_prev_us)
            loop_per_dac = max(0.999, min(1.001, loop_per_dac))
        return int(round(loop_ref_us + (dac_time_us - dac_ref_us) * loop_per_dac))

    def _get_current_playback_position_us(self) -> int:
        """Get the current playback position in server timestamp space."""
        return self._last_known_playback_position_us

    def get_timing_metrics(self) -> dict[str, float]:
        """Return current timing metrics for monitoring."""
        return {
            "playback_position_us": float(self._get_current_playback_position_us()),
            "buffered_audio_us": float(self._queued_duration_us),
            "dac_samples_recorded": len(self._dac_loop_calibrations),
        }

    def _log_chunk_timing(self, server_timestamp_us: int) -> None:
        """Log phase error and buffer status for debugging sync issues."""
        if self._phase_error_ema_init:
            now_us = int(self._loop.time() * 1_000_000)
            if now_us - self._last_phase_error_log_us >= 1_000_000:
                self._last_phase_error_log_us = now_us
                logger.debug(
                    "Phase error: %.1f ms, buffer: %.2f s",
                    self._phase_error_ema_us / 1000.0,
                    self._queued_duration_us / 1_000_000,
                )

    def _compute_minimum_buffer_duration(self) -> float:
        """Compute minimum buffer duration needed for network jitter.

        Returns the minimum buffer duration. The gap/overlap handling in
        submit() takes care of out-of-order chunks, so we use a fixed
        base buffer with the audio callback providing actual playback position.
        """
        # Base buffer: 200ms to start playback quickly while maintaining stability
        # With chunks arriving 5+ seconds early, we can afford aggressive buffering
        return 200_000

    def _update_correction_schedule(self, error_us: int) -> None:
        """Plan occasional sample drop/insert to correct phase drift.

        Positive error means DAC/server playback is ahead of our read cursor;
        schedule drops to catch up. Negative error means we're ahead; schedule
        inserts to slow down. Large errors trigger re-anchoring instead of
        aggressive correction to avoid artifacts.
        """
        if self._format is None or self._format.sample_rate <= 0:
            return

        # Smooth the error to avoid reacting to jitter
        if not self._phase_error_ema_init:
            self._phase_error_ema_us = float(error_us)
            self._phase_error_ema_init = True
        else:
            alpha = 0.90  # allow fresher error information for quicker response
            self._phase_error_ema_us = alpha * self._phase_error_ema_us + (1.0 - alpha) * float(
                error_us
            )

        abs_err = abs(self._phase_error_ema_us)

        # Deadband where we do nothing
        DEADBAND_US = 2_000  # 2 ms
        if abs_err <= DEADBAND_US:
            self._insert_every_n_frames = 0
            self._drop_every_n_frames = 0
            return

        # Re-anchor only if error is very large and cooldown has elapsed
        REANCHOR_THRESHOLD_US = 120_000  # 120 ms
        REANCHOR_COOLDOWN_US = 5_000_000  # 5 seconds
        now_loop_us = int(self._loop.time() * 1_000_000)
        if (
            abs_err > REANCHOR_THRESHOLD_US
            and not self._waiting_for_start
            and now_loop_us - self._last_reanchor_loop_time_us > REANCHOR_COOLDOWN_US
        ):
            logger.info("Phase error %.1f ms too large; re-anchoring", abs_err / 1000.0)
            # Reset cadence
            self._insert_every_n_frames = 0
            self._drop_every_n_frames = 0
            self._frames_until_next_insert = 0
            self._frames_until_next_drop = 0
            self._last_reanchor_loop_time_us = now_loop_us
            # Re-anchor on next chunk boundary by clearing queue
            self.clear()
            return

        # Convert error to equivalent frames
        frames_error = int(round(abs_err * self._format.sample_rate / 1_000_000.0))

        # Aim to correct within roughly 4 seconds with a max of ~50 corrections/s
        TARGET_SECONDS = 4.0
        max_corrections_per_sec = 50.0
        desired_corrections_per_sec = min(
            max_corrections_per_sec, frames_error / max(TARGET_SECONDS, 1.0)
        )

        interval_frames = int(self._format.sample_rate / max(desired_corrections_per_sec, 1.0))
        interval_frames = max(512, interval_frames)  # at least ~12 ms between corrections

        if self._phase_error_ema_us > 0:
            # We are behind (DAC ahead) -> drop to catch up
            self._drop_every_n_frames = interval_frames
            self._insert_every_n_frames = 0
        else:
            # We are ahead -> insert to slow down
            self._insert_every_n_frames = interval_frames
            self._drop_every_n_frames = 0

    def submit(self, server_timestamp_us: int, payload: bytes) -> None:
        """
        Queue an audio payload for playback, intelligently handling gaps and overlaps.

        Fills gaps with silence and trims overlaps to ensure a continuous stream.

        Args:
            server_timestamp_us: Server timestamp when this audio should play.
            payload: Raw PCM audio bytes.
        """
        if self._format is None:
            logger.debug("Audio format missing; dropping audio chunk")
            return
        if self._format.frame_size == 0:
            return
        if len(payload) % self._format.frame_size != 0:
            logger.warning(
                "Dropping audio chunk with invalid size: %s bytes (frame size %s)",
                len(payload),
                self._format.frame_size,
            )
            return

        # Calibrate DAC↔Loop time mapping from async context
        self._calibrate_dac_loop_mapping()

        now_us = int(self._loop.time() * 1_000_000)

        # On first real chunk, schedule start time aligned to server timeline
        if self._scheduled_start_loop_time_us is None:
            try:
                self._scheduled_start_loop_time_us = self._compute_client_time(server_timestamp_us)
            except Exception:
                self._scheduled_start_loop_time_us = int(self._loop.time() * 1_000_000)
            # Best-effort DAC schedule; refined later as calibrations accumulate
            est_dac = self._estimate_dac_time_for_server_timestamp(server_timestamp_us)
            # Only set DAC time when we can estimate it; otherwise use loop-based gating
            self._scheduled_start_dac_time_us = est_dac if est_dac else None
            self._waiting_for_start = True
            self._first_server_timestamp_us = server_timestamp_us
            # If scheduled start is very near now, suspect unsynchronized fallback mapping
            if self._scheduled_start_loop_time_us - now_us <= 700_000:
                self._early_start_suspect = True

        # While waiting to start, keep the scheduled loop start updated as time sync improves
        elif self._waiting_for_start and self._first_server_timestamp_us is not None:
            try:
                updated_loop_start = self._compute_client_time(self._first_server_timestamp_us)
                # Only update if it moves significantly (> 5ms) to avoid churn
                if abs(updated_loop_start - (self._scheduled_start_loop_time_us or 0)) > 5_000:
                    self._scheduled_start_loop_time_us = updated_loop_start
                    est_dac = self._estimate_dac_time_for_server_timestamp(
                        self._first_server_timestamp_us
                    )
                    self._scheduled_start_dac_time_us = est_dac if est_dac else None
            except Exception:
                pass

        # If we started too early due to fallback mapping, re-anchor once sync improves
        elif not self._waiting_for_start and self._early_start_suspect and not self._has_reanchored:
            # Heuristic: when DAC mapping becomes available or loop mapping pushes
            # the start > 1s into the future for current server timestamp, re-anchor.
            est_dac = self._estimate_dac_time_for_server_timestamp(server_timestamp_us)
            loop_start_now = None
            try:
                loop_start_now = self._compute_client_time(server_timestamp_us)
            except Exception:
                loop_start_now = None

            if est_dac or (loop_start_now is not None and loop_start_now - now_us > 1_000_000):
                # Clear current queue and reset timing; then continue to queue this chunk
                logger.info("Re-anchoring playback after time sync matured")
                self.clear()
                try:
                    self._scheduled_start_loop_time_us = self._compute_client_time(
                        server_timestamp_us
                    )
                except Exception:
                    self._scheduled_start_loop_time_us = now_us
                est_dac2 = self._estimate_dac_time_for_server_timestamp(server_timestamp_us)
                self._scheduled_start_dac_time_us = est_dac2 if est_dac2 else None
                self._waiting_for_start = True
                self._first_server_timestamp_us = server_timestamp_us
                self._has_reanchored = True

        # After calibration, if we have both a DAC-derived playback position and a
        # server-timeline cursor, compute phase error and schedule micro-corrections.
        if self._last_known_playback_position_us > 0 and self._server_ts_cursor_us > 0:
            phase_error_us = self._last_known_playback_position_us - self._server_ts_cursor_us
            self._update_correction_schedule(phase_error_us)

        # Log timing information (verbose, for debugging latency issues)
        self._log_chunk_timing(server_timestamp_us)

        # Initialize expected next timestamp on first chunk
        if self._expected_next_timestamp is None:
            self._expected_next_timestamp = server_timestamp_us
        # Handle gap: insert silence to fill the gap
        elif server_timestamp_us > self._expected_next_timestamp:
            gap_us = server_timestamp_us - self._expected_next_timestamp
            gap_frames = (gap_us * self._format.sample_rate) // 1_000_000
            silence_bytes = gap_frames * self._format.frame_size
            silence = b"\x00" * silence_bytes
            self._queue.put_nowait(
                _QueuedChunk(
                    server_timestamp_us=self._expected_next_timestamp,
                    audio_data=silence,
                )
            )
            # Account for inserted silence in buffer duration
            silence_duration_us = (gap_frames * 1_000_000) // self._format.sample_rate
            self._queued_duration_us += silence_duration_us
            logger.debug(
                "Gap: %.1f ms filled with silence",
                gap_us / 1000.0,
            )
            self._expected_next_timestamp = server_timestamp_us

        # Handle overlap: trim the start of the chunk
        elif server_timestamp_us < self._expected_next_timestamp:
            overlap_us = self._expected_next_timestamp - server_timestamp_us
            overlap_frames = (overlap_us * self._format.sample_rate) // 1_000_000
            trim_bytes = overlap_frames * self._format.frame_size
            if trim_bytes < len(payload):
                payload = payload[trim_bytes:]
                server_timestamp_us = self._expected_next_timestamp
                logger.debug(
                    "Overlap: %.1f ms trimmed",
                    overlap_us / 1000.0,
                )
            else:
                # Entire chunk is overlap, skip it
                logger.debug(
                    "Overlap: %.1f ms (chunk skipped, already played)",
                    overlap_us / 1000.0,
                )
                return

        # Queue the chunk
        if len(payload) > 0:
            # Compute duration from the post-trim payload
            chunk_frames = len(payload) // self._format.frame_size
            chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
            chunk = _QueuedChunk(
                server_timestamp_us=server_timestamp_us,
                audio_data=payload,
            )
            self._queue.put_nowait(chunk)
            # Track duration of queued audio
            self._queued_duration_us += chunk_duration_us

        # Update expected position for next chunk
        if len(payload) > 0:
            self._expected_next_timestamp = server_timestamp_us + chunk_duration_us

        # Compute minimum buffer needed for network jitter
        min_duration_us = self._compute_minimum_buffer_duration()

        queue_size = self._queue.qsize()
        if self._queued_duration_us >= min_duration_us and not self._stream_started:
            if self._stream is not None:
                self._stream.start()
                self._stream_started = True
                logger.info(
                    "✓ Stream STARTED: %d chunks, %.2f seconds buffered, "
                    "ready to play (min required: %.2f s)",
                    queue_size,
                    self._queued_duration_us / 1_000_000,
                    min_duration_us / 1_000_000,
                )
        elif not self._stream_started:
            buffered_pct = (
                100 * self._queued_duration_us / min_duration_us if min_duration_us > 0 else 0
            )
            logger.debug(
                "Buffering: %.2f/%.2f s (%.0f%%)",
                self._queued_duration_us / 1_000_000,
                min_duration_us / 1_000_000,
                buffered_pct,
            )

    def _close_stream(self) -> None:
        """Close the audio output stream."""
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Failed to close audio output stream")
        self._stream = None
