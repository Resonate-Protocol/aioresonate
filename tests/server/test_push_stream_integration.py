"""End-to-end integration tests for PushStream.

These tests exercise the full audio streaming flow from prepare_audio
through to player receipt, including late joiner catch-up.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from aiosendspin.models.core import ClientHelloPayload, StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import ChannelRouter
from aiosendspin.server.player_state import PlayerRecord, PlayerRegistry
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import PlayerRole

# Standard test audio format: 48kHz stereo 16-bit
TEST_AUDIO_FORMAT = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)


def get_binary_messages(messages: list) -> list[bytes]:
    """Filter binary (bytes) messages from a list of sent messages."""
    return [m for m in messages if isinstance(m, bytes)]


def get_stream_start_messages(messages: list) -> list[StreamStartMessage]:
    """Filter StreamStartMessage from a list of sent messages."""
    return [m for m in messages if isinstance(m, StreamStartMessage)]


class TestFullStreamingFlow:
    """Integration tests for complete audio streaming scenarios."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop with controllable time."""
        loop = MagicMock()
        loop.time.return_value = 0.0
        return loop

    @pytest.fixture
    def player_registry(self, mock_loop: MagicMock) -> PlayerRegistry:
        """Create a real player registry."""
        return PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)

    @pytest.fixture
    def channel_router(self) -> ChannelRouter:
        """Create a real channel router."""
        return ChannelRouter()

    @pytest.fixture
    def push_stream(
        self,
        mock_loop: MagicMock,
        player_registry: PlayerRegistry,
        channel_router: ChannelRouter,
    ) -> PushStream:
        """Create a PushStream for testing."""
        return PushStream(
            loop=mock_loop,
            group_id="test-group",
            player_registry=player_registry,
            channel_router=channel_router,
        )

    def _create_mock_connection(self, client_id: str) -> MagicMock:
        """Create a mock player connection with message tracking."""
        conn = MagicMock()
        conn.client_id = client_id
        conn.closing = False
        conn.info = ClientHelloPayload(
            client_id=client_id,
            name=f"Test Player {client_id}",
            version=1,
            supported_roles=[Roles.PLAYER.value],
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM,
                        sample_rate=48000,
                        bit_depth=16,
                        channels=2,
                    ),
                ],
                buffer_capacity=100_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
        )
        # Track all messages (both JSON objects and binary bytes)
        conn.sent_messages: list = []

        def track_message(msg: object) -> None:
            conn.sent_messages.append(msg)

        conn.send_message.side_effect = track_message

        # Binary path: PlayerRole uses try_send_binary for droppable audio chunks
        def track_binary(data: bytes) -> bool:
            conn.sent_messages.append(data)
            return True

        conn.try_send_binary = MagicMock(side_effect=track_binary)
        # Non-blocking players check queue_high_water - default to False (not full)
        conn.queue_high_water = MagicMock(return_value=False)
        return conn

    def _register_player(
        self,
        player_registry: PlayerRegistry,
        client_id: str,
    ) -> tuple[PlayerRecord, MagicMock]:
        """Register a player and attach a mock connection with PlayerRole."""
        record = player_registry.get_or_create(client_id)
        record.group_id = "test-group"
        conn = self._create_mock_connection(client_id)
        record.connection = conn
        # Create and attach PlayerRole for the connection
        player_role = PlayerRole(_record=record, _connection=conn)
        player_role.on_connect()
        record.player_role = player_role
        return record, conn

    @pytest.mark.asyncio
    async def test_single_player_receives_audio(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """A single connected player receives stream/start and audio chunks."""
        # Setup: Register one player
        _, conn = self._register_player(player_registry, "player-1")

        # Act: Push audio
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Assert: Player received stream/start message
        stream_starts = get_stream_start_messages(conn.sent_messages)
        assert len(stream_starts) == 1

        # Assert: Player received binary audio chunk
        binary_msgs = get_binary_messages(conn.sent_messages)
        assert len(binary_msgs) >= 1
        assert len(binary_msgs[0]) > 0  # Has header + audio data

    @pytest.mark.asyncio
    async def test_multiple_players_receive_same_audio(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Multiple players all receive the same audio chunks."""
        # Setup: Register three players
        _, conn1 = self._register_player(player_registry, "player-1")
        _, conn2 = self._register_player(player_registry, "player-2")
        _, conn3 = self._register_player(player_registry, "player-3")

        # Act: Push audio
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Assert: All players received stream/start
        for conn in [conn1, conn2, conn3]:
            stream_starts = get_stream_start_messages(conn.sent_messages)
            assert len(stream_starts) == 1

        # Assert: All players received the same binary data
        binary1 = get_binary_messages(conn1.sent_messages)
        binary2 = get_binary_messages(conn2.sent_messages)
        binary3 = get_binary_messages(conn3.sent_messages)
        assert binary1 == binary2 == binary3

    @pytest.mark.asyncio
    async def test_late_joiner_receives_catchup(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """A player joining mid-stream receives cached audio."""
        # Setup: Register first player
        self._register_player(player_registry, "player-1")

        # Act: Push initial audio
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Verify cache has chunks
        assert push_stream.has_cached_chunks()

        # Late joiner arrives (time stays at 0 - chunks are in future)
        _, conn2 = self._register_player(player_registry, "player-2")
        push_stream.on_player_join("player-2")

        # Assert: Late joiner received stream/start
        stream_starts = get_stream_start_messages(conn2.sent_messages)
        assert len(stream_starts) == 1

        # Assert: Late joiner received cached audio chunk
        binary_msgs = get_binary_messages(conn2.sent_messages)
        assert len(binary_msgs) >= 1

    @pytest.mark.asyncio
    async def test_multiple_commits_maintain_timing(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Multiple commits produce chunks with contiguous timestamps."""
        # Setup
        _, conn = self._register_player(player_registry, "player-1")

        # Act: Push multiple chunks
        for _ in range(3):
            push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
            await push_stream.commit_audio()

        # Assert: Received 3 binary chunks
        binary_msgs = get_binary_messages(conn.sent_messages)
        assert len(binary_msgs) == 3

        # Verify timestamps are advancing (check header bytes)
        # Binary format: 1 byte type + 8 byte timestamp (big-endian signed) + data
        timestamps = []
        for binary in binary_msgs:
            # Skip message type byte, read 8-byte timestamp (big-endian)
            timestamp = int.from_bytes(binary[1:9], byteorder="big", signed=True)
            timestamps.append(timestamp)

        # Timestamps should be increasing
        assert timestamps[1] > timestamps[0]
        assert timestamps[2] > timestamps[1]

    @pytest.mark.asyncio
    async def test_stop_sends_stream_end(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Stopping the stream sends stream/end to all players."""
        # Setup
        _, conn = self._register_player(player_registry, "player-1")

        # Push some audio first
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Clear tracked messages to focus on stop
        conn.sent_messages.clear()

        # Act: Stop the stream
        push_stream.stop()

        # Assert: Player received stream/end
        assert len(conn.sent_messages) == 1
        assert conn.sent_messages[0].type == "stream/end"

    @pytest.mark.asyncio
    async def test_clear_resets_stream_state(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Clearing the stream resets timing and cache."""
        # Setup
        _, conn = self._register_player(player_registry, "player-1")

        # Push audio
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Clear messages
        conn.sent_messages.clear()

        # Act: Clear the stream
        push_stream.clear()

        # Assert: Player received stream/clear
        assert len(conn.sent_messages) == 1
        assert conn.sent_messages[0].type == "stream/clear"

        # Verify cache is empty
        assert not push_stream.has_cached_chunks()

        # Assert: Next commit will send new stream/start
        conn.sent_messages.clear()
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        stream_starts = get_stream_start_messages(conn.sent_messages)
        assert len(stream_starts) == 1

    @pytest.mark.asyncio
    async def test_disconnected_player_skipped(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Disconnected players don't receive audio."""
        # Setup: Two players, one connected, one disconnected
        _, conn1 = self._register_player(player_registry, "player-1")
        record2, conn2 = self._register_player(player_registry, "player-2")

        # Disconnect player-2
        record2.connection = None

        # Act: Push audio
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Assert: Only player-1 received data
        assert len(conn1.sent_messages) >= 2  # stream/start + audio chunk

        # Player-2's connection mock shouldn't have been called
        # (it's disconnected, so send_message wasn't invoked)
        assert len(conn2.sent_messages) == 0


class TestMultiChannelStreaming:
    """Integration tests for multi-channel audio streaming."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 0.0
        return loop

    @pytest.fixture
    def player_registry(self, mock_loop: MagicMock) -> PlayerRegistry:
        """Create a real player registry."""
        return PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)

    @pytest.fixture
    def channel_router(self) -> ChannelRouter:
        """Create a real channel router."""
        return ChannelRouter()

    @pytest.fixture
    def push_stream(
        self,
        mock_loop: MagicMock,
        player_registry: PlayerRegistry,
        channel_router: ChannelRouter,
    ) -> PushStream:
        """Create a PushStream for testing."""
        return PushStream(
            loop=mock_loop,
            group_id="test-group",
            player_registry=player_registry,
            channel_router=channel_router,
        )

    def _create_mock_connection(self, client_id: str) -> MagicMock:
        """Create a mock player connection."""
        conn = MagicMock()
        conn.client_id = client_id
        conn.closing = False
        conn.info = ClientHelloPayload(
            client_id=client_id,
            name=f"Test Player {client_id}",
            version=1,
            supported_roles=[Roles.PLAYER.value],
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM,
                        sample_rate=48000,
                        bit_depth=16,
                        channels=2,
                    ),
                ],
                buffer_capacity=100_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
        )
        conn.sent_messages: list = []

        def _send_message(msg: object) -> None:
            conn.sent_messages.append(msg)

        def _try_send_binary(data: bytes) -> bool:
            conn.sent_messages.append(data)
            return True

        conn.send_message.side_effect = _send_message
        conn.try_send_binary = MagicMock(side_effect=_try_send_binary)
        # Non-blocking players check queue_high_water - default to False (not full)
        conn.queue_high_water = MagicMock(return_value=False)
        return conn

    @pytest.mark.asyncio
    async def test_players_receive_their_channel_audio(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        channel_router: ChannelRouter,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Players on different channels receive their assigned audio."""
        # Setup: Two players on different channels
        channel_a = uuid4()
        channel_b = uuid4()

        record1 = player_registry.get_or_create("player-1")
        record1.group_id = "test-group"
        conn1 = self._create_mock_connection("player-1")
        record1.connection = conn1
        role1 = PlayerRole(_record=record1, _connection=conn1)
        role1.on_connect()
        record1.player_role = role1
        channel_router.set_channel("player-1", channel_a)

        record2 = player_registry.get_or_create("player-2")
        record2.group_id = "test-group"
        conn2 = self._create_mock_connection("player-2")
        record2.connection = conn2
        role2 = PlayerRole(_record=record2, _connection=conn2)
        role2.on_connect()
        record2.player_role = role2
        channel_router.set_channel("player-2", channel_b)

        # Create different audio for each channel (modify content)
        audio_a = pcm_48000_stereo_16bit
        audio_b = bytes([0xFF] * len(pcm_48000_stereo_16bit))

        # Act: Push audio to both channels
        push_stream.prepare_audio(audio_a, TEST_AUDIO_FORMAT, channel_id=channel_a)
        push_stream.prepare_audio(audio_b, TEST_AUDIO_FORMAT, channel_id=channel_b)
        await push_stream.commit_audio()

        # Assert: Each player received audio from their channel
        binary1 = get_binary_messages(conn1.sent_messages)
        binary2 = get_binary_messages(conn2.sent_messages)
        assert len(binary1) >= 1
        assert len(binary2) >= 1

        # The binary data should be different (different source audio)
        assert binary1[0] != binary2[0]

    @pytest.mark.asyncio
    async def test_main_channel_default(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Players without explicit channel assignment receive main channel."""
        # Setup: Player with no channel assignment
        record = player_registry.get_or_create("player-1")
        record.group_id = "test-group"
        conn = self._create_mock_connection("player-1")
        record.connection = conn
        player_role = PlayerRole(_record=record, _connection=conn)
        player_role.on_connect()
        record.player_role = player_role

        # Act: Push audio to main channel (default)
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Assert: Player received the audio
        binary_msgs = get_binary_messages(conn.sent_messages)
        assert len(binary_msgs) >= 1


class TestBackpressureIntegration:
    """Integration tests for backpressure handling."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 0.0
        return loop

    @pytest.fixture
    def player_registry(self, mock_loop: MagicMock) -> PlayerRegistry:
        """Create a player registry with small buffer capacity."""
        return PlayerRegistry(loop=mock_loop, default_buffer_capacity=10_000)

    @pytest.fixture
    def push_stream(
        self,
        mock_loop: MagicMock,
        player_registry: PlayerRegistry,
    ) -> PushStream:
        """Create a PushStream for testing."""
        return PushStream(
            loop=mock_loop,
            group_id="test-group",
            player_registry=player_registry,
            channel_router=ChannelRouter(),
        )

    def _create_mock_connection(self, client_id: str) -> MagicMock:
        """Create a mock player connection."""
        conn = MagicMock()
        conn.client_id = client_id
        conn.closing = False
        conn.info = ClientHelloPayload(
            client_id=client_id,
            name=f"Test Player {client_id}",
            version=1,
            supported_roles=[Roles.PLAYER.value],
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM,
                        sample_rate=48000,
                        bit_depth=16,
                        channels=2,
                    ),
                ],
                buffer_capacity=100_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
        )
        conn.sent_messages: list = []

        def _send_message(msg: object) -> None:
            conn.sent_messages.append(msg)

        def _try_send_binary(data: bytes) -> bool:
            conn.sent_messages.append(data)
            return True

        conn.send_message.side_effect = _send_message
        conn.try_send_binary = MagicMock(side_effect=_try_send_binary)
        # Non-blocking players check queue_high_water - default to False (not full)
        conn.queue_high_water = MagicMock(return_value=False)
        return conn

    @pytest.mark.asyncio
    async def test_buffer_tracker_updated_on_send(
        self,
        push_stream: PushStream,
        player_registry: PlayerRegistry,
        pcm_48000_stereo_16bit: bytes,
    ) -> None:
        """Sending audio updates the player's buffer tracker."""
        # Setup
        record = player_registry.get_or_create("player-1")
        record.group_id = "test-group"
        conn = self._create_mock_connection("player-1")
        record.connection = conn
        # Create and attach PlayerRole
        player_role = PlayerRole(_record=record, _connection=conn)
        player_role.on_connect()
        record.player_role = player_role

        # Initial state: buffer tracker is empty
        assert record.buffer_tracker is not None
        initial_bytes = record.buffer_tracker.buffered_bytes

        # Act: Push audio
        push_stream.prepare_audio(pcm_48000_stereo_16bit, TEST_AUDIO_FORMAT)
        await push_stream.commit_audio()

        # Assert: Buffer tracker now has bytes registered
        assert record.buffer_tracker.buffered_bytes > initial_bytes
