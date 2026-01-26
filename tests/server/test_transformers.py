"""Tests for AudioTransformer protocol and implementations."""

from __future__ import annotations

from aiosendspin.server.transformers import (
    AudioTransformer,
    FlacEncoder,
    PcmPassthrough,
    TransformerPool,
)


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

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                pass

        # Should be recognized as implementing the protocol
        transformer: AudioTransformer = ValidTransformer()
        assert transformer.process(b"test", 0, 1000) == [b"test"]

    def test_protocol_defines_get_header_method(self) -> None:
        """AudioTransformer requires get_header() method."""

        class TransformerWithHeader:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> list[bytes]:
                return [pcm]

            def flush(self) -> list[bytes]:
                return []

            def get_header(self) -> bytes | None:
                return b"header"

            def reset(self) -> None:
                pass

        transformer: AudioTransformer = TransformerWithHeader()
        assert transformer.get_header() == b"header"

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

            def get_header(self) -> bytes | None:
                return None

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

            def get_header(self) -> bytes | None:
                return None

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

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                pass

        transformer: AudioTransformer = TransformerWithFlush()
        assert transformer.flush() == [b"final"]


class TestPcmPassthrough:
    """Tests for PcmPassthrough transformer."""

    def test_passthrough_returns_input_unchanged(self) -> None:
        """PcmPassthrough returns input PCM data unchanged."""
        transformer = PcmPassthrough()
        pcm = b"\x00\x01\x02\x03"
        result = transformer.process(pcm, timestamp_us=0, duration_us=1000)
        assert result == pcm

    def test_passthrough_has_no_header(self) -> None:
        """PcmPassthrough has no codec header."""
        transformer = PcmPassthrough()
        assert transformer.get_header() is None

    def test_passthrough_reset_is_noop(self) -> None:
        """PcmPassthrough reset is a no-op."""
        transformer = PcmPassthrough()
        transformer.reset()  # Should not raise

    def test_passthrough_accepts_kwargs(self) -> None:
        """PcmPassthrough accepts and ignores format parameters."""
        # Should accept and ignore format params for TransformerPool compatibility
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.process(b"x", 0, 1000) == b"x"


class TestTransformerPool:
    """Tests for TransformerPool."""

    def test_get_or_create_creates_new_transformer(self) -> None:
        """Pool creates new transformer when none exists for key."""
        pool = TransformerPool()
        transformer = pool.get_or_create(
            PcmPassthrough, sample_rate=48000, bit_depth=16, channels=2
        )
        assert isinstance(transformer, PcmPassthrough)

    def test_get_or_create_returns_same_instance(self) -> None:
        """Pool returns same instance for identical key."""
        pool = TransformerPool()
        t1 = pool.get_or_create(PcmPassthrough, sample_rate=48000, bit_depth=16, channels=2)
        t2 = pool.get_or_create(PcmPassthrough, sample_rate=48000, bit_depth=16, channels=2)
        assert t1 is t2

    def test_get_or_create_different_config_different_instance(self) -> None:
        """Pool creates different instances for different keys."""
        pool = TransformerPool()
        t1 = pool.get_or_create(PcmPassthrough, sample_rate=48000, bit_depth=16, channels=2)
        t2 = pool.get_or_create(PcmPassthrough, sample_rate=44100, bit_depth=16, channels=2)
        assert t1 is not t2

    def test_reset_all_calls_reset_on_all_transformers(self) -> None:
        """Pool reset_all calls reset on every transformer."""
        reset_counts: list[int] = []

        class CountingTransformer:
            def __init__(self, **_kwargs: object) -> None:
                self.index = len(reset_counts)
                reset_counts.append(0)

            def process(self, pcm: bytes, _ts: int, _dur: int) -> bytes:
                return pcm

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                reset_counts[self.index] += 1

        pool = TransformerPool()
        pool.get_or_create(
            CountingTransformer,
            sample_rate=48000,
            bit_depth=16,
            channels=2,  # type: ignore[type-var]
        )
        pool.get_or_create(
            CountingTransformer,
            sample_rate=44100,
            bit_depth=16,
            channels=2,  # type: ignore[type-var]
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
        total_output = bytearray()
        for i in range(4):
            result = encoder.process(pcm, timestamp_us=i * 25000, duration_us=25000)
            total_output.extend(result)
        assert len(total_output) > 0

    def test_flac_encoder_has_header(self) -> None:
        """FlacEncoder produces fLaC header."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)
        encoder.process(pcm, timestamp_us=0, duration_us=25000)
        header = encoder.get_header()
        assert header is not None
        assert header.startswith(b"fLaC")

    def test_flac_encoder_reset_clears_state(self) -> None:
        """FlacEncoder reset clears internal state."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)
        encoder.process(pcm, timestamp_us=0, duration_us=25000)
        encoder.reset()
        assert encoder._initialized is False  # noqa: SLF001
        assert encoder._codec_header is None  # noqa: SLF001
