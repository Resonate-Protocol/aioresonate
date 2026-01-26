"""Tests for PipelineManager encoding infrastructure."""

from __future__ import annotations

from uuid import UUID

import pytest

from aiosendspin.models import AudioCodec
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.pipeline import EncodedChunk, PipelineKey, PipelineManager


class TestPipelineKey:
    """Tests for PipelineKey structure."""

    def test_pipeline_key_has_channel_id(self) -> None:
        """PipelineKey should have channel_id field."""
        key = PipelineKey(
            channel_id=MAIN_CHANNEL,
            source_format=AudioFormat(sample_rate=44100, bit_depth=16, channels=2),
            target_format=AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
            codec=AudioCodec.FLAC,
        )
        assert key.channel_id == MAIN_CHANNEL

    def test_pipeline_key_is_hashable(self) -> None:
        """PipelineKey should be hashable for use as dict key."""
        key = PipelineKey(
            channel_id=MAIN_CHANNEL,
            source_format=AudioFormat(sample_rate=44100, bit_depth=16, channels=2),
            target_format=AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
            codec=AudioCodec.FLAC,
        )
        # Should not raise
        hash(key)
        d = {key: "test"}
        assert d[key] == "test"


class TestEncodedChunk:
    """Tests for EncodedChunk structure."""

    def test_encoded_chunk_has_required_fields(self) -> None:
        """EncodedChunk should have timestamp_us, data, byte_count, sample_count, duration_us."""
        chunk = EncodedChunk(
            timestamp_us=123,
            data=b"\x00\x01\x02\x03",
            byte_count=4,
            sample_count=100,
            duration_us=2500,
        )
        assert chunk.timestamp_us == 123
        assert chunk.data == b"\x00\x01\x02\x03"
        assert chunk.byte_count == 4
        assert chunk.sample_count == 100
        assert chunk.duration_us == 2500


class TestPipelineManagerConstruction:
    """Tests for PipelineManager construction."""

    def test_creates_instance(self) -> None:
        """PipelineManager should be creatable."""
        manager = PipelineManager()
        assert manager is not None


class TestPipelineManagerAddPipeline:
    """Tests for adding pipelines."""

    @pytest.fixture
    def manager(self) -> PipelineManager:
        """Create a PipelineManager for testing."""
        return PipelineManager()

    @pytest.fixture
    def source_format(self) -> AudioFormat:
        """Source PCM format."""
        return AudioFormat(sample_rate=44100, bit_depth=16, channels=2)

    @pytest.fixture
    def target_format_flac(self) -> AudioFormat:
        """Target FLAC format."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    def test_add_pipeline_returns_key(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_flac: AudioFormat,
    ) -> None:
        """add_pipeline should return a PipelineKey."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )
        assert isinstance(key, PipelineKey)
        assert key.channel_id == MAIN_CHANNEL
        assert key.source_format == source_format
        assert key.target_format == target_format_flac
        assert key.codec == AudioCodec.FLAC

    def test_add_same_pipeline_twice_returns_same_key(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_flac: AudioFormat,
    ) -> None:
        """Adding same (channel, source, target, codec) twice should return same key (dedup)."""
        key1 = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )
        key2 = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )
        assert key1 == key2

    def test_different_channels_create_different_pipelines(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_flac: AudioFormat,
    ) -> None:
        """Different channel_ids should create different pipelines."""
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        key1 = manager.add_pipeline(
            channel_id=channel_a,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )
        key2 = manager.add_pipeline(
            channel_id=channel_b,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )
        assert key1 != key2

    def test_encoders_not_shared_across_channels(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_flac: AudioFormat,
    ) -> None:
        """
        Encoders must not be shared across independent streams (channels).

        Audio encoders are stream-stateful; sharing one encoder instance across two
        channels can corrupt output.
        """
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        manager.add_pipeline(
            channel_id=channel_a,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )
        manager.add_pipeline(
            channel_id=channel_b,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )

        # Private state assertion: we should have one encoder per channel for the same codec/params.
        assert len(manager._encoders) == 2  # noqa: SLF001


class TestPipelineManagerProcess:
    """Tests for processing audio through pipelines."""

    @pytest.fixture
    def manager(self) -> PipelineManager:
        """Create a PipelineManager for testing."""
        return PipelineManager()

    @pytest.fixture
    def source_format(self) -> AudioFormat:
        """Source PCM format."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    @pytest.fixture
    def target_format_pcm(self) -> AudioFormat:
        """Target PCM format (no encoding)."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    @pytest.fixture
    def target_format_flac(self) -> AudioFormat:
        """Target FLAC format."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    def test_process_returns_dict_of_encoded_chunks(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """Process should return dict[PipelineKey, list[EncodedChunk]]."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_pcm,
            codec=AudioCodec.PCM,
        )

        # 25ms of stereo 16-bit 48kHz = 1200 samples * 4 bytes = 4800 bytes
        pcm = bytes(4800)
        prepared = {MAIN_CHANNEL: (pcm, source_format, 0)}

        result = manager.process(prepared, {key})

        assert isinstance(result, dict)
        assert key in result
        assert isinstance(result[key], list)
        assert len(result[key]) > 0
        assert all(isinstance(chunk, EncodedChunk) for chunk in result[key])

    def test_process_only_requested_pipelines(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
        target_format_flac: AudioFormat,
    ) -> None:
        """Process should only encode requested pipelines."""
        key_pcm = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_pcm,
            codec=AudioCodec.PCM,
        )
        key_flac = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_flac,
            codec=AudioCodec.FLAC,
        )

        pcm = bytes(4800)
        prepared = {MAIN_CHANNEL: (pcm, source_format, 0)}

        # Only request PCM pipeline
        result = manager.process(prepared, {key_pcm})

        assert key_pcm in result
        assert key_flac not in result

    def test_process_encoded_chunk_has_correct_byte_count(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """EncodedChunk.byte_count should match len(data)."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_pcm,
            codec=AudioCodec.PCM,
        )

        pcm = bytes(4800)
        prepared = {MAIN_CHANNEL: (pcm, source_format, 0)}

        result = manager.process(prepared, {key})

        for chunk in result[key]:
            assert chunk.byte_count == len(chunk.data)

    def test_process_encoded_chunk_has_duration_us(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format_pcm: AudioFormat,
    ) -> None:
        """EncodedChunk should have positive duration_us."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format_pcm,
            codec=AudioCodec.PCM,
        )

        pcm = bytes(4800)
        prepared = {MAIN_CHANNEL: (pcm, source_format, 0)}

        result = manager.process(prepared, {key})

        for chunk in result[key]:
            assert chunk.duration_us > 0


class TestPipelineManagerCodecHeader:
    """Tests for codec header retrieval."""

    @pytest.fixture
    def manager(self) -> PipelineManager:
        """Create a PipelineManager for testing."""
        return PipelineManager()

    @pytest.fixture
    def source_format(self) -> AudioFormat:
        """Source PCM format."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    def test_get_codec_header_returns_none_for_pcm(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
    ) -> None:
        """get_codec_header should return None for PCM pipelines."""
        target_pcm = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_pcm,
            codec=AudioCodec.PCM,
        )

        header = manager.get_codec_header(key)
        assert header is None

    def test_get_codec_header_returns_bytes_for_flac(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
    ) -> None:
        """get_codec_header should return bytes for FLAC pipelines."""
        target_flac = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_flac,
            codec=AudioCodec.FLAC,
        )

        header = manager.get_codec_header(key)
        assert header is not None
        assert isinstance(header, bytes)
        # FLAC header should start with "fLaC"
        assert header.startswith(b"fLaC")


class TestPipelineManagerRemoveAndReset:
    """Tests for removing pipelines and resetting."""

    @pytest.fixture
    def manager(self) -> PipelineManager:
        """Create a PipelineManager for testing."""
        return PipelineManager()

    @pytest.fixture
    def source_format(self) -> AudioFormat:
        """Source PCM format."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    @pytest.fixture
    def target_format(self) -> AudioFormat:
        """Target PCM format."""
        return AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    def test_remove_pipeline(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> None:
        """remove_pipeline should remove the pipeline."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format,
            codec=AudioCodec.PCM,
        )

        manager.remove_pipeline(key)

        # Adding again should create a new pipeline (since old was removed)
        # This is verified by checking internal state
        assert not manager.has_pipeline(key)

    def test_remove_nonexistent_pipeline_is_noop(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> None:
        """remove_pipeline on nonexistent key should not raise."""
        key = PipelineKey(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format,
            codec=AudioCodec.PCM,
        )
        # Should not raise
        manager.remove_pipeline(key)

    def test_reset_clears_all_pipelines(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> None:
        """Reset should clear all pipelines."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format,
            codec=AudioCodec.PCM,
        )

        manager.reset()

        assert not manager.has_pipeline(key)

    def test_has_pipeline(
        self,
        manager: PipelineManager,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> None:
        """has_pipeline should return True for existing pipelines."""
        key = manager.add_pipeline(
            channel_id=MAIN_CHANNEL,
            source_format=source_format,
            target_format=target_format,
            codec=AudioCodec.PCM,
        )

        assert manager.has_pipeline(key) is True

        fake_key = PipelineKey(
            channel_id=UUID("11111111-1111-1111-1111-111111111111"),
            source_format=source_format,
            target_format=target_format,
            codec=AudioCodec.PCM,
        )
        assert manager.has_pipeline(fake_key) is False
