"""Tests for AudioTransformer protocol and implementations."""

from __future__ import annotations

from aiosendspin.server.transformers import AudioTransformer


class TestAudioTransformerProtocol:
    """Tests for AudioTransformer protocol."""

    def test_protocol_defines_process_method(self) -> None:
        """AudioTransformer requires process() method."""

        class ValidTransformer:
            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> bytes:
                return pcm

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                pass

        # Should be recognized as implementing the protocol
        transformer: AudioTransformer = ValidTransformer()
        assert transformer.process(b"test", 0, 1000) == b"test"

    def test_protocol_defines_get_header_method(self) -> None:
        """AudioTransformer requires get_header() method."""

        class TransformerWithHeader:
            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> bytes:
                return pcm

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

            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> bytes:
                return pcm

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                self.reset_count += 1

        transformer = ResettableTransformer()
        transformer.reset()
        assert transformer.reset_count == 1
