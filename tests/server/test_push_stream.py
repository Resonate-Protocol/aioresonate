"""Tests for PushStream push-based audio streaming API."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import AudioCodec
from aiosendspin.server.channels import MAIN_CHANNEL, ChannelRouter
from aiosendspin.server.player_state import PlayerRegistry
from aiosendspin.server.push_stream import PushStream
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
