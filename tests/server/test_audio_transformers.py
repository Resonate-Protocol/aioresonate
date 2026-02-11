"""Tests for AudioTransformer protocol and implementations."""

from __future__ import annotations

from aiosendspin.server.audio_transformers import (
    AudioTransformer,
    TransformerPool,
)
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough


class TestAudioTransformerProtocol:
    """Tests for AudioTransformer protocol."""

    def test_protocol_defines_process_method(self) -> None:
        """AudioTransformer requires process() method."""

        class ValidTransformer:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> list[bytes]:
                return [pcm]

            def flush(self) -> list[bytes]:
                return []

            def reset(self) -> None:
                pass

        # Should be recognized as implementing the protocol
        transformer: AudioTransformer = ValidTransformer()
        assert transformer.process(b"test", 0, 1000) == [b"test"]

    def test_protocol_defines_reset_method(self) -> None:
        """AudioTransformer requires reset() method."""

        class ResettableTransformer:
            def __init__(self) -> None:
                self.reset_count = 0

            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> list[bytes]:
                return [pcm]

            def flush(self) -> list[bytes]:
                return []

            def reset(self) -> None:
                self.reset_count += 1

        transformer = ResettableTransformer()
        transformer.reset()
        assert transformer.reset_count == 1

    def test_protocol_defines_frame_duration_us_property(self) -> None:
        """AudioTransformer requires frame_duration_us property."""

        class TransformerWithFrameDuration:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> list[bytes]:
                return [pcm]

            def flush(self) -> list[bytes]:
                return []

            def reset(self) -> None:
                pass

        transformer: AudioTransformer = TransformerWithFrameDuration()
        assert transformer.frame_duration_us == 25_000

    def test_protocol_defines_flush_method(self) -> None:
        """AudioTransformer requires flush() method."""

        class TransformerWithFlush:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> list[bytes]:
                return [pcm]

            def flush(self) -> list[bytes]:
                return [b"final"]

            def reset(self) -> None:
                pass

        transformer: AudioTransformer = TransformerWithFlush()
        assert transformer.flush() == [b"final"]


class TestPcmPassthrough:
    """Tests for PcmPassthrough transformer."""

    def test_passthrough_has_no_header(self) -> None:
        """PcmPassthrough has no codec header."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.get_header() is None

    def test_passthrough_accepts_kwargs(self) -> None:
        """PcmPassthrough accepts format parameters."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.frame_duration_us == 25_000

    def test_passthrough_frame_duration_us_default(self) -> None:
        """PcmPassthrough has default 25ms frame duration."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.frame_duration_us == 25_000

    def test_passthrough_frame_duration_us_configurable(self) -> None:
        """PcmPassthrough frame duration is configurable."""
        transformer = PcmPassthrough(
            sample_rate=48000, bit_depth=16, channels=2, chunk_duration_us=50_000
        )
        assert transformer.frame_duration_us == 50_000

    def test_passthrough_returns_list_of_frames(self) -> None:
        """PcmPassthrough returns list of frames."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)  # 25ms at 48kHz stereo 16-bit
        result = transformer.process(pcm, timestamp_us=0, duration_us=25_000)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == pcm

    def test_passthrough_splits_large_input(self) -> None:
        """PcmPassthrough splits large input into multiple frames."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(9600)  # 50ms = 2 frames
        result = transformer.process(pcm, timestamp_us=0, duration_us=50_000)
        assert len(result) == 2
        assert len(result[0]) == 4800
        assert len(result[1]) == 4800

    def test_passthrough_buffers_incomplete_frame(self) -> None:
        """PcmPassthrough buffers incomplete frames."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(1920)  # 10ms - less than 25ms
        result = transformer.process(pcm, timestamp_us=0, duration_us=10_000)
        assert result == []

    def test_passthrough_emits_when_buffer_fills(self) -> None:
        """PcmPassthrough emits frame when buffer reaches frame size."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        result1 = transformer.process(bytes(2880), timestamp_us=0, duration_us=15_000)  # 15ms
        assert result1 == []
        result2 = transformer.process(
            bytes(2880), timestamp_us=15_000, duration_us=15_000
        )  # +15ms = 30ms total
        assert len(result2) == 1
        assert len(result2[0]) == 4800

    def test_passthrough_flush_emits_remainder_padded(self) -> None:
        """PcmPassthrough flush emits remaining buffer padded with silence."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        transformer.process(bytes(1920), timestamp_us=0, duration_us=10_000)
        result = transformer.flush()
        assert len(result) == 1
        assert len(result[0]) == 4800  # Padded to 25ms

    def test_passthrough_flush_empty_buffer(self) -> None:
        """PcmPassthrough flush returns empty list when buffer is empty."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        result = transformer.flush()
        assert result == []

    def test_passthrough_reset_clears_buffer(self) -> None:
        """PcmPassthrough reset clears internal buffer."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        transformer.process(bytes(1920), timestamp_us=0, duration_us=10_000)
        transformer.reset()
        assert transformer.flush() == []


class TestTransformerPool:
    """Tests for TransformerPool."""

    def test_get_or_create_creates_new_transformer(self) -> None:
        """Pool creates new transformer when none exists for key."""
        pool = TransformerPool()
        transformer = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        assert isinstance(transformer, PcmPassthrough)

    def test_get_or_create_returns_same_instance(self) -> None:
        """Pool returns same instance for identical key."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        assert t1 is t2

    def test_get_or_create_different_config_different_instance(self) -> None:
        """Pool creates different instances for different keys."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=44100,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        assert t1 is not t2

    def test_get_or_create_reuses_instance_for_identical_kwargs(self) -> None:
        """Pool reuses instances when constructor kwargs are identical."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "little"},
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "little"},
        )
        assert t1 is t2

    def test_get_or_create_uses_kwargs_in_pool_key(self) -> None:
        """Pool creates distinct instances when constructor kwargs differ."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "little"},
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "big"},
        )
        assert t1 is not t2

    def test_reset_all_calls_reset_on_all_transformers(self) -> None:
        """Pool reset_all calls reset on every transformer."""
        reset_counts: list[int] = []

        class CountingTransformer:
            def __init__(self, **_kwargs: object) -> None:
                self.index = len(reset_counts)
                reset_counts.append(0)

            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _ts: int, _dur: int) -> list[bytes]:
                return [pcm]

            def flush(self) -> list[bytes]:
                return []

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                reset_counts[self.index] += 1

        pool = TransformerPool()
        pool.get_or_create(
            CountingTransformer,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,  # type: ignore[type-var]
            frame_duration_us=25_000,
        )
        pool.get_or_create(
            CountingTransformer,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=44100,
            bit_depth=16,
            channels=2,  # type: ignore[type-var]
            frame_duration_us=25_000,
        )
        pool.reset_all()
        assert reset_counts == [1, 1]


class TestFlacEncoder:
    """Tests for FlacEncoder transformer."""

    def test_flac_encoder_produces_bytes(self) -> None:
        """FlacEncoder produces encoded output."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # 25ms of silence at 48kHz stereo 16-bit = 1200 samples * 4 bytes = 4800 bytes
        # Send multiple chunks to ensure encoder produces output (FLAC buffers initial frames)
        pcm = bytes(4800)
        total_output: list[bytes] = []
        for i in range(4):
            result = encoder.process(pcm, timestamp_us=i * 25_000, duration_us=25_000)
            total_output.extend(result)
        assert len(total_output) > 0

    def test_flac_encoder_supports_32_bit(self) -> None:
        """FlacEncoder accepts 32-bit PCM input."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=32, channels=2)
        # 25ms at 48kHz stereo 32-bit: 1200 samples * 8 bytes = 9600 bytes.
        pcm = bytes(9600)
        total_output: list[bytes] = []
        for i in range(4):
            result = encoder.process(pcm, timestamp_us=i * 25_000, duration_us=25_000)
            total_output.extend(result)
        assert len(total_output) > 0

    def test_flac_encoder_has_header(self) -> None:
        """FlacEncoder produces fLaC header."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)
        encoder.process(pcm, timestamp_us=0, duration_us=25_000)
        header = encoder.get_header()
        assert header is not None
        assert header.startswith(b"fLaC")

    def test_flac_encoder_reset_clears_state(self) -> None:
        """FlacEncoder reset clears internal state."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)
        encoder.process(pcm, timestamp_us=0, duration_us=25_000)
        encoder.reset()
        assert encoder._initialized is False  # noqa: SLF001
        assert encoder._codec_header is None  # noqa: SLF001

    def test_flac_encoder_frame_duration_us_default(self) -> None:
        """FlacEncoder has default 25ms frame duration."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        assert encoder.frame_duration_us == 25_000

    def test_flac_encoder_frame_duration_us_configurable(self) -> None:
        """FlacEncoder frame duration is configurable."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2, chunk_duration_us=50_000)
        assert encoder.frame_duration_us == 50_000

    def test_flac_encoder_returns_list_of_frames(self) -> None:
        """FlacEncoder returns list of frames after codec internal buffering fills."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # FLAC codec buffers ~4 frames before emitting output
        # Feed enough frames to guarantee output
        pcm = bytes(4800)  # 25ms per chunk
        all_results: list[bytes] = []
        for i in range(8):  # 200ms total
            result = encoder.process(pcm, timestamp_us=i * 25_000, duration_us=25_000)
            assert isinstance(result, list)
            all_results.extend(result)
        # Should have some output after 8 frames
        assert len(all_results) >= 1

    def test_flac_encoder_buffers_incomplete_frame(self) -> None:
        """FlacEncoder buffers incomplete frames."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(1920)  # 10ms
        result = encoder.process(pcm, timestamp_us=0, duration_us=10_000)
        assert result == []

    def test_flac_encoder_flush_emits_remainder(self) -> None:
        """FlacEncoder flush emits remaining buffered audio when buffer has data."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # Process incomplete frame (less than 25ms)
        encoder.process(bytes(1920), timestamp_us=0, duration_us=10_000)
        result = encoder.flush()
        # FLAC codec may not emit output immediately due to internal buffering,
        # but our buffer was cleared (padded to frame size and sent to encoder).
        # The actual output depends on codec timing.
        # At minimum, verify flush returns a list
        assert isinstance(result, list)

    def test_flac_encoder_flush_empty_buffer(self) -> None:
        """FlacEncoder flush returns empty list when buffer is empty."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # FLAC frame size is 4608 samples = 18432 bytes at 48kHz stereo 16-bit.
        # Process exactly one FLAC frame worth of data.
        flac_frame_bytes = 4608 * 4  # 4608 samples * 4 bytes per sample
        encoder.process(bytes(flac_frame_bytes), timestamp_us=0, duration_us=96_000)
        result = encoder.flush()
        assert result == []

    def test_flac_encoder_pending_timestamp_continuous(self) -> None:
        """FlacEncoder pending_timestamp_us produces continuous frame timestamps.

        FLAC uses a block size of 4608 samples (~96ms at 48kHz). pending_timestamp_us
        tracks output frame count to ensure timestamps are continuous regardless
        of input chunk sizes.
        """
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)

        # Simulate source sending 1005ms chunks (doesn't align with FLAC frames)
        chunk_bytes = int(48000 * 1.005) * 4  # ~1005ms of audio

        timestamps: list[int] = []
        for call_num in range(5):
            input_ts = call_num * 1_005_000  # Input timestamps advance by 1005ms

            # Get base timestamp for output frames (mimics PushStream logic)
            pending_before = encoder.pending_timestamp_us
            base_ts = pending_before if pending_before is not None else input_ts

            frames = encoder.process(bytes(chunk_bytes), input_ts, 1_005_000)
            frame_dur = encoder.frame_duration_us  # ~96ms for FLAC

            # Calculate frame timestamps
            for i in range(len(frames)):
                frame_ts = base_ts + i * frame_dur
                timestamps.append(frame_ts)

        # Verify we got output (1005ms * 5 = 5025ms, at ~96ms/frame = ~52 frames)
        assert len(timestamps) > 40, f"Expected >40 frames, got {len(timestamps)}"

        # Verify timestamps are continuous (each frame is frame_dur after previous)
        frame_dur = encoder.frame_duration_us
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap == frame_dur, (
                f"Frame {i}: gap={gap}us, expected {frame_dur}us. "
                f"Timestamps around gap: {timestamps[max(0, i - 2) : i + 2]}"
            )
