"""Tests for PushStream push-based audio streaming API."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import AudioCodec
from aiosendspin.server.channels import MAIN_CHANNEL, ChannelRouter
from aiosendspin.server.player_state import PlayerRegistry
from aiosendspin.server.push_stream import (
    DurationMismatchError,
    PushStream,
    StreamStoppedError,
)
from aiosendspin.server.stream import AudioFormat


class TestPushStreamConstruction:
    """Tests for PushStream construction."""

    def test_creates_instance_with_required_args(self, mock_loop: MagicMock) -> None:
        """PushStream should be creatable with required arguments."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        assert stream is not None


class TestPushStreamAPIShape:
    """Tests for PushStream API method signatures."""

    @pytest.fixture
    def push_stream(self, mock_loop: MagicMock) -> PushStream:
        """Create a PushStream for testing."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()
        return PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

    def test_prepare_audio_exists_and_is_sync(self, push_stream: PushStream) -> None:
        """prepare_audio should exist and be synchronous."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm = bytes(4800)  # 25ms of silence

        # Should not raise, should be synchronous (not a coroutine)
        result = push_stream.prepare_audio(pcm, fmt)
        assert not asyncio.iscoroutine(result)

    def test_prepare_audio_accepts_channel_id(self, push_stream: PushStream) -> None:
        """prepare_audio should accept optional channel_id."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm = bytes(4800)
        custom_channel = UUID("11111111-1111-1111-1111-111111111111")

        # Should not raise
        push_stream.prepare_audio(pcm, fmt, channel_id=custom_channel)

    def test_prepare_audio_defaults_to_main_channel(self, push_stream: PushStream) -> None:
        """prepare_audio should default to MAIN_CHANNEL."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm = bytes(4800)

        # Should not raise - channel_id defaults to MAIN_CHANNEL
        push_stream.prepare_audio(pcm, fmt)

    @pytest.mark.asyncio
    async def test_commit_audio_exists_and_is_async(self, push_stream: PushStream) -> None:
        """commit_audio should exist and be asynchronous."""
        result = push_stream.commit_audio()
        assert asyncio.iscoroutine(result)

        # Await it to clean up
        play_start_us = await result
        assert isinstance(play_start_us, int)

    @pytest.mark.asyncio
    async def test_wait_for_buffer_space_exists_and_is_async(self, push_stream: PushStream) -> None:
        """wait_for_buffer_space should exist and be asynchronous."""
        result = push_stream.wait_for_buffer_space()
        assert asyncio.iscoroutine(result)

        # Await it to clean up
        await result

    def test_stop_exists_and_is_sync(self, push_stream: PushStream) -> None:
        """Stop should exist and be synchronous."""
        result = push_stream.stop()
        assert not asyncio.iscoroutine(result)

    def test_clear_exists_and_is_sync(self, push_stream: PushStream) -> None:
        """Clear should exist and be synchronous."""
        result = push_stream.clear()
        assert not asyncio.iscoroutine(result)

    def test_is_stopped_property_exists(self, push_stream: PushStream) -> None:
        """is_stopped property should exist."""
        assert hasattr(push_stream, "is_stopped")
        assert isinstance(push_stream.is_stopped, bool)

    def test_is_stopped_initially_false(self, push_stream: PushStream) -> None:
        """is_stopped should be False initially."""
        assert push_stream.is_stopped is False

    def test_is_stopped_true_after_stop(self, push_stream: PushStream) -> None:
        """is_stopped should be True after stop() is called."""
        push_stream.stop()
        assert push_stream.is_stopped is True


class TestPrepareAudio:
    """Tests for prepare_audio behavior and pending audio tracking."""

    @pytest.fixture
    def push_stream(self, mock_loop: MagicMock) -> PushStream:
        """Create a PushStream for testing."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()
        return PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

    def test_has_pending_audio_false_initially(self, push_stream: PushStream) -> None:
        """has_pending_audio should return False when nothing is prepared."""
        assert push_stream.has_pending_audio() is False

    def test_has_pending_audio_true_after_prepare(self, push_stream: PushStream) -> None:
        """has_pending_audio should return True after prepare_audio."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm = bytes(4800)

        push_stream.prepare_audio(pcm, fmt)

        assert push_stream.has_pending_audio() is True

    def test_prepare_stores_pcm_for_channel(self, push_stream: PushStream) -> None:
        """prepare_audio should store PCM data for the specified channel."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm = b"\x00\x01\x02\x03" * 100

        push_stream.prepare_audio(pcm, fmt, channel_id=MAIN_CHANNEL)

        # Access internal state to verify
        pending = push_stream.get_pending_audio()
        assert MAIN_CHANNEL in pending
        stored_pcm, stored_fmt = pending[MAIN_CHANNEL]
        assert stored_pcm == pcm
        assert stored_fmt == fmt

    def test_prepare_twice_replaces_not_appends(self, push_stream: PushStream) -> None:
        """Calling prepare_audio twice for same channel should replace, not append."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm1 = b"\x00\x01\x02\x03" * 100
        pcm2 = b"\x04\x05\x06\x07" * 50

        push_stream.prepare_audio(pcm1, fmt, channel_id=MAIN_CHANNEL)
        push_stream.prepare_audio(pcm2, fmt, channel_id=MAIN_CHANNEL)

        pending = push_stream.get_pending_audio()
        stored_pcm, _ = pending[MAIN_CHANNEL]
        # Should be pcm2, not pcm1 + pcm2
        assert stored_pcm == pcm2
        assert len(stored_pcm) == len(pcm2)

    def test_prepare_different_channels_stored_separately(self, push_stream: PushStream) -> None:
        """Calling prepare_audio for different channels should store separately."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm1 = b"\x00\x01\x02\x03" * 100
        pcm2 = b"\x04\x05\x06\x07" * 50
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        push_stream.prepare_audio(pcm1, fmt, channel_id=channel_a)
        push_stream.prepare_audio(pcm2, fmt, channel_id=channel_b)

        pending = push_stream.get_pending_audio()
        assert len(pending) == 2
        assert pending[channel_a][0] == pcm1
        assert pending[channel_b][0] == pcm2

    @pytest.mark.asyncio
    async def test_commit_clears_pending_audio(self, push_stream: PushStream) -> None:
        """commit_audio should clear pending audio after commit."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC)
        pcm = bytes(4800)

        push_stream.prepare_audio(pcm, fmt)
        assert push_stream.has_pending_audio() is True

        await push_stream.commit_audio()

        assert push_stream.has_pending_audio() is False


class TestCommitAudio:
    """Tests for commit_audio core logic."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop with time()."""
        loop = MagicMock()
        loop.time.return_value = 1000.0  # 1000 seconds = 1_000_000_000 us
        return loop

    @pytest.fixture
    def push_stream(self, mock_loop: MagicMock) -> PushStream:
        """Create a PushStream for testing."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()
        return PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

    @pytest.fixture
    def source_format(self) -> AudioFormat:
        """Source PCM format (48kHz stereo 16-bit)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    @pytest.mark.asyncio
    async def test_first_commit_initializes_play_start_us(
        self, push_stream: PushStream, source_format: AudioFormat
    ) -> None:
        """First commit should initialize play_start_us to now + initial_delay."""
        # 25ms of audio at 48kHz stereo 16-bit = 1200 samples * 4 bytes = 4800 bytes
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)

        play_start_us = await push_stream.commit_audio()

        # Should be loop.time() in microseconds plus some initial delay
        # loop.time() = 1000.0 seconds = 1_000_000_000 us
        assert play_start_us > 1_000_000_000
        # And reasonable (within 1 second of initial delay)
        assert play_start_us < 1_001_000_000

    @pytest.mark.asyncio
    async def test_commit_returns_play_start_us(
        self, push_stream: PushStream, source_format: AudioFormat
    ) -> None:
        """commit_audio should return play_start_us."""
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)

        result = await push_stream.commit_audio()

        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.asyncio
    async def test_subsequent_commits_advance_timing(
        self, push_stream: PushStream, source_format: AudioFormat
    ) -> None:
        """Subsequent commits should advance play_start_us by audio duration."""
        pcm = bytes(4800)  # 25ms

        push_stream.prepare_audio(pcm, source_format)
        first_start = await push_stream.commit_audio()

        push_stream.prepare_audio(pcm, source_format)
        second_start = await push_stream.commit_audio()

        # Second commit should start 25ms (25000 us) after first
        expected_advance = 25000  # 25ms in microseconds
        assert second_start == first_start + expected_advance

    @pytest.mark.asyncio
    async def test_commit_raises_stream_stopped_error_when_stopped(
        self, push_stream: PushStream, source_format: AudioFormat
    ) -> None:
        """commit_audio should raise StreamStoppedError if stream is stopped."""
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        push_stream.stop()

        with pytest.raises(StreamStoppedError):
            await push_stream.commit_audio()

    @pytest.mark.asyncio
    async def test_commit_validates_duration_alignment(self, push_stream: PushStream) -> None:
        """commit_audio should raise if channel durations don't match."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        # 25ms on channel A
        pcm_25ms = bytes(4800)
        # 50ms on channel B (different duration)
        pcm_50ms = bytes(9600)

        push_stream.prepare_audio(pcm_25ms, fmt, channel_id=channel_a)
        push_stream.prepare_audio(pcm_50ms, fmt, channel_id=channel_b)

        with pytest.raises(DurationMismatchError):
            await push_stream.commit_audio()

    @pytest.mark.asyncio
    async def test_commit_allows_small_duration_differences(self, push_stream: PushStream) -> None:
        """commit_audio should allow small rounding differences in duration."""
        fmt_48k = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)
        fmt_44k = AudioFormat(sample_rate=44100, bit_depth=16, channels=2, codec=AudioCodec.PCM)
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        # Both approximately 25ms but slightly different sample counts
        # 48000 * 0.025 = 1200 samples * 4 bytes = 4800 bytes
        pcm_48k = bytes(4800)
        # 44100 * 0.025 = 1102.5 -> 1102 samples * 4 bytes = 4408 bytes
        pcm_44k = bytes(4408)

        push_stream.prepare_audio(pcm_48k, fmt_48k, channel_id=channel_a)
        push_stream.prepare_audio(pcm_44k, fmt_44k, channel_id=channel_b)

        # Should not raise - durations are close enough
        await push_stream.commit_audio()

    @pytest.mark.asyncio
    async def test_commit_with_no_pending_audio_is_noop(self, push_stream: PushStream) -> None:
        """commit_audio with no pending audio should not raise."""
        # Should not raise, returns 0 or initial timing
        result = await push_stream.commit_audio()
        assert isinstance(result, int)


class TestBackpressure:
    """Tests for backpressure and timeline shift."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop with time()."""
        loop = MagicMock()
        loop.time.return_value = 1000.0  # 1000 seconds = 1_000_000_000 us
        return loop

    @pytest.fixture
    def source_format(self) -> AudioFormat:
        """Source PCM format (48kHz stereo 16-bit)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    def _create_mock_player(self, client_id: str, wait_us: int, mock_loop: MagicMock) -> MagicMock:
        """Create a mock player with a buffer tracker that returns specified wait."""
        player = MagicMock()
        player.client_id = client_id
        player.is_connected = True
        player.buffer_tracker = MagicMock()
        player.buffer_tracker.time_until_capacity.return_value = wait_us
        player.buffer_tracker.loop = mock_loop
        player.preferred_format = None  # Uses default
        return player

    @pytest.mark.asyncio
    async def test_no_shift_when_all_players_have_capacity(
        self, mock_loop: MagicMock, source_format: AudioFormat
    ) -> None:
        """If all players have capacity, timeline should not shift."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        # Create player with no wait needed
        player = self._create_mock_player("player-1", wait_us=0, mock_loop=mock_loop)
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        first_start = await push_stream.commit_audio()

        push_stream.prepare_audio(pcm, source_format)
        second_start = await push_stream.commit_audio()

        # No shift - second starts exactly 25ms after first
        assert second_start == first_start + 25000

    @pytest.mark.asyncio
    async def test_shift_when_player_needs_wait(
        self, mock_loop: MagicMock, source_format: AudioFormat
    ) -> None:
        """If player needs wait_us, timeline should shift forward."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        # Create player that needs 10ms wait
        player = self._create_mock_player("player-1", wait_us=10_000, mock_loop=mock_loop)
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        first_start = await push_stream.commit_audio()

        # Update mock to need 10ms wait for second commit
        player.buffer_tracker.time_until_capacity.return_value = 10_000

        push_stream.prepare_audio(pcm, source_format)
        second_start = await push_stream.commit_audio()

        # Should be shifted: 25ms audio + 10ms wait = 35ms after first
        assert second_start == first_start + 35000

    @pytest.mark.asyncio
    async def test_shift_uses_max_wait_across_players(
        self, mock_loop: MagicMock, source_format: AudioFormat
    ) -> None:
        """Multiple slow players should use max wait time."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        # Create players with different wait times
        player1 = self._create_mock_player("player-1", wait_us=5_000, mock_loop=mock_loop)
        player2 = self._create_mock_player("player-2", wait_us=15_000, mock_loop=mock_loop)
        player3 = self._create_mock_player("player-3", wait_us=10_000, mock_loop=mock_loop)
        registry._players["player-1"] = player1  # noqa: SLF001
        registry._players["player-2"] = player2  # noqa: SLF001
        registry._players["player-3"] = player3  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        first_start = await push_stream.commit_audio()

        push_stream.prepare_audio(pcm, source_format)
        second_start = await push_stream.commit_audio()

        # Should use max wait (15ms): 25ms audio + 15ms wait = 40ms after first
        assert second_start == first_start + 40000

    @pytest.mark.asyncio
    async def test_no_shift_with_no_connected_players(
        self, mock_loop: MagicMock, source_format: AudioFormat
    ) -> None:
        """No connected players means no backpressure."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        first_start = await push_stream.commit_audio()

        push_stream.prepare_audio(pcm, source_format)
        second_start = await push_stream.commit_audio()

        # No shift - second starts exactly 25ms after first
        assert second_start == first_start + 25000
