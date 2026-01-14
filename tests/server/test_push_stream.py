"""Tests for PushStream push-based audio streaming API."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import (
    BINARY_HEADER_SIZE,
    AudioCodec,
    BinaryMessageType,
    unpack_binary_header,
)
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
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


class TestSendChunks:
    """Tests for sending encoded chunks to players."""

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

    @pytest.fixture
    def target_format_pcm(self) -> AudioFormat:
        """Target PCM format (no encoding needed)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    def _create_mock_player(
        self,
        client_id: str,
        mock_loop: MagicMock,
        target_format: AudioFormat,
        *,
        is_connected: bool = True,
        wait_us: int = 0,
    ) -> MagicMock:
        """Create a mock player with connection and buffer tracker."""
        player = MagicMock()
        player.client_id = client_id
        player.is_connected = is_connected
        player.preferred_format = target_format
        player.buffer_tracker = MagicMock()
        player.buffer_tracker.time_until_capacity.return_value = wait_us
        player.buffer_tracker.register = MagicMock()
        player.buffer_tracker.loop = mock_loop
        player.connection = MagicMock()
        player.connection.send_message = MagicMock()
        return player

    @pytest.mark.asyncio
    async def test_connected_player_receives_chunks(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """After commit, connected players should receive chunks via send_message."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # 25ms of audio
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Player should have received at least one chunk
        assert player.connection.send_message.called

    @pytest.mark.asyncio
    async def test_chunk_has_correct_timestamp_in_header(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Chunk binary header should contain correct timestamp."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        play_start_us = await push_stream.commit_audio()

        # Get the sent message
        sent_data = player.connection.send_message.call_args[0][0]
        assert isinstance(sent_data, bytes)

        # Unpack and verify header
        header = unpack_binary_header(sent_data)
        assert header.message_type == BinaryMessageType.AUDIO_CHUNK.value
        assert header.timestamp_us == play_start_us

    @pytest.mark.asyncio
    async def test_buffer_tracker_register_called(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Player's buffer_tracker.register() should be called with end_time and byte_count."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # buffer_tracker.register should have been called
        assert player.buffer_tracker.register.called
        # First arg is end_time_us, second is byte_count
        end_time_us, byte_count = player.buffer_tracker.register.call_args[0]
        assert end_time_us > 0
        assert byte_count > 0

    @pytest.mark.asyncio
    async def test_player_gets_chunks_from_assigned_channel(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Player should get chunks from their assigned channel."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        # Create two channels with different audio
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        player_a = self._create_mock_player(
            "player-a", mock_loop, target_format_pcm, is_connected=True
        )
        player_b = self._create_mock_player(
            "player-b", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-a"] = player_a  # noqa: SLF001
        registry._players["player-b"] = player_b  # noqa: SLF001

        # Assign players to different channels
        router.set_channel("player-a", channel_a)
        router.set_channel("player-b", channel_b)

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Prepare different audio for each channel
        pcm_a = b"\x01\x02" * 2400  # 25ms, distinct pattern
        pcm_b = b"\x03\x04" * 2400  # 25ms, distinct pattern

        push_stream.prepare_audio(pcm_a, source_format, channel_id=channel_a)
        push_stream.prepare_audio(pcm_b, source_format, channel_id=channel_b)
        await push_stream.commit_audio()

        # Both players should receive chunks
        assert player_a.connection.send_message.called
        assert player_b.connection.send_message.called

        # The content should be different (after header)
        data_a = player_a.connection.send_message.call_args[0][0][BINARY_HEADER_SIZE:]
        data_b = player_b.connection.send_message.call_args[0][0][BINARY_HEADER_SIZE:]
        # PCM output differs based on input
        assert data_a != data_b

    @pytest.mark.asyncio
    async def test_multiple_chunks_have_contiguous_timestamps(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """If pipeline produces multiple chunks, timestamps should be contiguous."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Prepare more audio to ensure multiple chunks (50ms = 2x 25ms chunks)
        source_format = AudioFormat(
            sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM
        )
        pcm = bytes(9600)  # 50ms
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Get all sent messages (filter out non-binary messages like stream/start)
        calls = player.connection.send_message.call_args_list
        binary_calls = [call for call in calls if isinstance(call[0][0], bytes)]
        if len(binary_calls) > 1:
            # Extract timestamps from each chunk
            timestamps = []
            for call in binary_calls:
                data = call[0][0]
                header = unpack_binary_header(data)
                timestamps.append(header.timestamp_us)

            # Verify timestamps are in ascending order
            for i in range(1, len(timestamps)):
                assert timestamps[i] > timestamps[i - 1]

    @pytest.mark.asyncio
    async def test_disconnected_player_does_not_receive(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Disconnected players should not receive chunks (no error)."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=False
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)

        # Should not raise even with disconnected player
        await push_stream.commit_audio()

        # Disconnected player should not receive messages
        assert not player.connection.send_message.called


class TestStreamStart:
    """Tests for stream/start message sending."""

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

    @pytest.fixture
    def target_format_pcm(self) -> AudioFormat:
        """Target PCM format (no encoding needed)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    def _create_mock_player(
        self,
        client_id: str,
        mock_loop: MagicMock,
        target_format: AudioFormat,
        *,
        is_connected: bool = True,
    ) -> MagicMock:
        """Create a mock player with connection and buffer tracker."""
        player = MagicMock()
        player.client_id = client_id
        player.is_connected = is_connected
        player.preferred_format = target_format
        player.buffer_tracker = MagicMock()
        player.buffer_tracker.time_until_capacity.return_value = 0
        player.buffer_tracker.register = MagicMock()
        player.buffer_tracker.loop = mock_loop
        player.connection = MagicMock()
        player.connection.send_message = MagicMock()
        return player

    @pytest.mark.asyncio
    async def test_first_chunk_triggers_stream_start(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """First chunk to player should trigger stream/start before audio."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Should have received at least two messages: stream/start and audio chunk
        calls = player.connection.send_message.call_args_list
        assert len(calls) >= 2

        # First message should be stream/start (JSON message)
        first_msg = calls[0][0][0]
        assert isinstance(first_msg, StreamStartMessage)

    @pytest.mark.asyncio
    async def test_stream_start_contains_correct_format(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """stream/start should contain correct format info."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Get the stream/start message
        first_msg = player.connection.send_message.call_args_list[0][0][0]
        assert isinstance(first_msg, StreamStartMessage)

        # Verify format info
        assert first_msg.payload.player is not None
        assert first_msg.payload.player.codec == target_format_pcm.codec
        assert first_msg.payload.player.sample_rate == target_format_pcm.sample_rate
        assert first_msg.payload.player.channels == target_format_pcm.channels
        assert first_msg.payload.player.bit_depth == target_format_pcm.bit_depth

    @pytest.mark.asyncio
    async def test_stream_start_includes_codec_header_for_flac(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
    ) -> None:
        """stream/start should include codec_header for FLAC."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        target_format_flac = AudioFormat(
            sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.FLAC
        )

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_flac, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Use enough audio to ensure FLAC encoder produces output
        # FLAC encoder buffers internally, needs ~100ms to emit first packet
        # 100ms at 48kHz stereo 16-bit = 4800 samples * 4 bytes = 19200 bytes
        pcm = bytes(19200)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Get the stream/start message
        calls = player.connection.send_message.call_args_list
        assert len(calls) > 0, "Player should receive at least stream/start"
        first_msg = calls[0][0][0]
        assert isinstance(first_msg, StreamStartMessage)

        # Verify codec header is present for FLAC
        assert first_msg.payload.player is not None
        assert first_msg.payload.player.codec_header is not None
        # FLAC header should be base64 encoded
        assert len(first_msg.payload.player.codec_header) > 0

    @pytest.mark.asyncio
    async def test_subsequent_chunks_dont_resend_stream_start(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Subsequent chunks should not re-send stream/start."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # First commit
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Count stream/start messages
        stream_start_count_1 = sum(
            1
            for call in player.connection.send_message.call_args_list
            if isinstance(call[0][0], StreamStartMessage)
        )
        assert stream_start_count_1 == 1

        # Second commit
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Count should still be 1
        stream_start_count_2 = sum(
            1
            for call in player.connection.send_message.call_args_list
            if isinstance(call[0][0], StreamStartMessage)
        )
        assert stream_start_count_2 == 1


class TestStopClear:
    """Tests for stop() and clear() methods."""

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

    @pytest.fixture
    def target_format_pcm(self) -> AudioFormat:
        """Target PCM format (no encoding needed)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    def _create_mock_player(
        self,
        client_id: str,
        mock_loop: MagicMock,
        target_format: AudioFormat,
        *,
        is_connected: bool = True,
    ) -> MagicMock:
        """Create a mock player with connection and buffer tracker."""
        player = MagicMock()
        player.client_id = client_id
        player.is_connected = is_connected
        player.preferred_format = target_format
        player.buffer_tracker = MagicMock()
        player.buffer_tracker.time_until_capacity.return_value = 0
        player.buffer_tracker.register = MagicMock()
        player.buffer_tracker.reset = MagicMock()
        player.buffer_tracker.loop = mock_loop
        player.connection = MagicMock()
        player.connection.send_message = MagicMock()
        return player

    # Tests for stop()

    @pytest.mark.asyncio
    async def test_stop_sets_is_stopped(self, mock_loop: MagicMock) -> None:
        """stop() should set is_stopped = True."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        assert not push_stream.is_stopped
        push_stream.stop()
        assert push_stream.is_stopped

    @pytest.mark.asyncio
    async def test_stop_sends_stream_end_message(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """stop() should send StreamEndMessage to connected players."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        push_stream.stop()

        # Player should have received stream/end message
        calls = player.connection.send_message.call_args_list
        assert len(calls) == 1
        msg = calls[0][0][0]
        assert isinstance(msg, StreamEndMessage)

    @pytest.mark.asyncio
    async def test_commit_raises_after_stop(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
    ) -> None:
        """commit_audio() should raise StreamStoppedError after stop."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        push_stream.stop()

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)

        with pytest.raises(StreamStoppedError):
            await push_stream.commit_audio()

    # Tests for clear()

    def test_clear_resets_channel_buffers(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
    ) -> None:
        """clear() should reset channel buffers."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        assert push_stream.has_pending_audio()

        push_stream.clear()
        assert not push_stream.has_pending_audio()

    @pytest.mark.asyncio
    async def test_clear_resets_timing(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
    ) -> None:
        """clear() should reset timing (next_chunk_start_us)."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Commit to initialize timing
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        first_start = await push_stream.commit_audio()

        # Clear should reset timing
        push_stream.clear()

        # Advance mock time
        mock_loop.time.return_value = 2000.0

        # Next commit should reinitialize timing at new time
        push_stream.prepare_audio(pcm, source_format)
        new_start = await push_stream.commit_audio()

        # Should be based on new time (2000s), not continued from first commit
        expected_new_start = int(2000.0 * 1_000_000) + 100_000  # 100ms initial delay
        assert new_start == expected_new_start
        assert new_start != first_start + 25000  # Not just continuation

    @pytest.mark.asyncio
    async def test_clear_sends_stream_clear_message(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """clear() should send StreamClearMessage to connected players."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        push_stream.clear()

        # Player should have received stream/clear message
        calls = player.connection.send_message.call_args_list
        assert len(calls) == 1
        msg = calls[0][0][0]
        assert isinstance(msg, StreamClearMessage)

    @pytest.mark.asyncio
    async def test_clear_resets_buffer_trackers(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """clear() should reset player buffer trackers."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        push_stream.clear()

        # Buffer tracker reset should have been called
        player.buffer_tracker.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_resets_player_started_set(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """clear() should reset _player_started so stream/start is re-sent."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # First commit sends stream/start
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        stream_starts_before = sum(
            1
            for call in player.connection.send_message.call_args_list
            if isinstance(call[0][0], StreamStartMessage)
        )
        assert stream_starts_before == 1

        # Clear resets _player_started
        push_stream.clear()

        # Next commit should send stream/start again
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        stream_starts_after = sum(
            1
            for call in player.connection.send_message.call_args_list
            if isinstance(call[0][0], StreamStartMessage)
        )
        assert stream_starts_after == 2  # Two stream/start messages total


class TestWaitForBufferSpace:
    """Tests for wait_for_buffer_space behavior."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop with time()."""
        loop = MagicMock()
        loop.time.return_value = 1000.0  # 1000 seconds = 1_000_000_000 us
        return loop

    @pytest.fixture
    def target_format_pcm(self) -> AudioFormat:
        """Target PCM format (no encoding needed)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    def _create_mock_player(
        self,
        client_id: str,
        mock_loop: MagicMock,
        target_format: AudioFormat,
        *,
        is_connected: bool = True,
        wait_us: int = 0,
    ) -> MagicMock:
        """Create a mock player with connection and buffer tracker."""
        player = MagicMock()
        player.client_id = client_id
        player.is_connected = is_connected
        player.preferred_format = target_format
        player.buffer_tracker = MagicMock()
        player.buffer_tracker.time_until_capacity.return_value = wait_us
        player.buffer_tracker.loop = mock_loop
        player.connection = MagicMock()
        return player

    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_connected_players(
        self,
        mock_loop: MagicMock,
    ) -> None:
        """wait_for_buffer_space should return immediately if no connected players."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Should return immediately without error
        await push_stream.wait_for_buffer_space()

    @pytest.mark.asyncio
    async def test_returns_immediately_if_all_players_have_capacity(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """wait_for_buffer_space should return immediately if all players have capacity."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True, wait_us=0
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Should return immediately without sleeping
        await push_stream.wait_for_buffer_space()
        # time_until_capacity should have been called
        player.buffer_tracker.time_until_capacity.assert_called()

    @pytest.mark.asyncio
    async def test_sleeps_for_max_wait_across_players(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """wait_for_buffer_space should sleep for max wait time across players."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        # Create players with different wait times
        player1 = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True, wait_us=5_000
        )
        player2 = self._create_mock_player(
            "player-2", mock_loop, target_format_pcm, is_connected=True, wait_us=15_000
        )
        player3 = self._create_mock_player(
            "player-3", mock_loop, target_format_pcm, is_connected=True, wait_us=10_000
        )
        registry._players["player-1"] = player1  # noqa: SLF001
        registry._players["player-2"] = player2  # noqa: SLF001
        registry._players["player-3"] = player3  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Track actual sleep duration
        slept_duration: float | None = None

        async def mock_sleep(duration: float) -> None:
            nonlocal slept_duration
            slept_duration = duration
            # Don't actually sleep in tests

        # Patch asyncio.sleep
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", mock_sleep)
            await push_stream.wait_for_buffer_space()

        # Should have slept for max wait (15ms = 0.015 seconds)
        assert slept_duration is not None
        assert slept_duration == pytest.approx(0.015, abs=0.001)

    @pytest.mark.asyncio
    async def test_uses_estimated_chunk_size(
        self,
        mock_loop: MagicMock,
        target_format_pcm: AudioFormat,
    ) -> None:
        """wait_for_buffer_space should use an estimated chunk size for capacity check."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True, wait_us=0
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        await push_stream.wait_for_buffer_space()

        # time_until_capacity should have been called with some byte estimate
        player.buffer_tracker.time_until_capacity.assert_called()
        call_args = player.buffer_tracker.time_until_capacity.call_args[0]
        assert len(call_args) >= 1
        byte_estimate = call_args[0]
        # Should be a reasonable estimate (not 0)
        assert byte_estimate > 0


class TestLateJoinerCache:
    """Tests for late joiner chunk cache."""

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

    @pytest.fixture
    def target_format_pcm(self) -> AudioFormat:
        """Target PCM format (no encoding needed)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    def _create_mock_player(
        self,
        client_id: str,
        mock_loop: MagicMock,
        target_format: AudioFormat,
        *,
        is_connected: bool = True,
    ) -> MagicMock:
        """Create a mock player with connection and buffer tracker."""
        player = MagicMock()
        player.client_id = client_id
        player.is_connected = is_connected
        player.preferred_format = target_format
        player.buffer_tracker = MagicMock()
        player.buffer_tracker.time_until_capacity.return_value = 0
        player.buffer_tracker.register = MagicMock()
        player.buffer_tracker.loop = mock_loop
        player.connection = MagicMock()
        player.connection.send_message = MagicMock()
        return player

    @pytest.mark.asyncio
    async def test_commit_caches_chunks(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """After commit, chunks should be cached."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Cache should have chunks
        assert push_stream.has_cached_chunks()

    @pytest.mark.asyncio
    async def test_cache_bounded_by_time_window(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Cache should be bounded by time window (old chunks pruned)."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Commit some audio
        pcm = bytes(4800)  # 25ms
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Advance time past the cache window (default 10 seconds)
        mock_loop.time.return_value = 2000.0  # 1000 seconds later

        # Commit more audio to trigger pruning
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Get catchup chunks for player's channel
        # Old chunks should be pruned
        chunks = push_stream.get_catchup_chunks("player-1")
        # All returned chunks should have recent timestamps
        for chunk in chunks:
            # Chunk timestamps should be >= now (2000.0s = 2_000_000_000 us)
            assert chunk.timestamp_us >= 2_000_000_000

    @pytest.mark.asyncio
    async def test_get_catchup_chunks_returns_for_player_channel(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """get_catchup_chunks should return chunks for player's channel."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        # Create two channels
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        player_a = self._create_mock_player(
            "player-a", mock_loop, target_format_pcm, is_connected=True
        )
        player_b = self._create_mock_player(
            "player-b", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-a"] = player_a  # noqa: SLF001
        registry._players["player-b"] = player_b  # noqa: SLF001

        router.set_channel("player-a", channel_a)
        router.set_channel("player-b", channel_b)

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Prepare different audio for each channel
        pcm_a = b"\x01\x02" * 2400
        pcm_b = b"\x03\x04" * 2400

        push_stream.prepare_audio(pcm_a, source_format, channel_id=channel_a)
        push_stream.prepare_audio(pcm_b, source_format, channel_id=channel_b)
        await push_stream.commit_audio()

        # Player A should get chunks from channel A
        chunks_a = push_stream.get_catchup_chunks("player-a")
        assert len(chunks_a) > 0

        # Player B should get chunks from channel B
        chunks_b = push_stream.get_catchup_chunks("player-b")
        assert len(chunks_b) > 0

    @pytest.mark.asyncio
    async def test_get_catchup_chunks_returns_future_chunks_only(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """get_catchup_chunks should return only chunks with timestamp >= now + margin."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        # Commit audio (timestamps will be around 1000s + 100ms initial delay)
        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        # Get current time in microseconds
        now_us = int(mock_loop.time() * 1_000_000)

        # All returned chunks should have timestamps >= now
        chunks = push_stream.get_catchup_chunks("player-1")
        for chunk in chunks:
            assert chunk.timestamp_us >= now_us

    @pytest.mark.asyncio
    async def test_clear_clears_cache(
        self,
        mock_loop: MagicMock,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """clear() should clear the chunk cache."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        router = ChannelRouter()

        player = self._create_mock_player(
            "player-1", mock_loop, target_format_pcm, is_connected=True
        )
        registry._players["player-1"] = player  # noqa: SLF001

        push_stream = PushStream(
            loop=mock_loop,
            player_registry=registry,
            channel_router=router,
        )

        pcm = bytes(4800)
        push_stream.prepare_audio(pcm, source_format)
        await push_stream.commit_audio()

        assert push_stream.has_cached_chunks()

        push_stream.clear()

        assert not push_stream.has_cached_chunks()
