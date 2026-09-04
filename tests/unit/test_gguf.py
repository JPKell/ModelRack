"""Tests for :mod:`modelrack.providers._gguf` — reading a GGUF header, hashing a file.

Every file here is written by the test through ``write_gguf`` in ``conftest.py``: the header is
the real format, the "weights" are a handful of bytes. The multi-gigabyte real files on the
reference machine are exercised by ``tests/live/test_llamacpp_live.py`` instead — a test that
hashes 40 GB is not a unit test, and one that needs them present cannot run in CI.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING

import pytest
from baseaicore import ValidationError, normalize_digest

from modelrack.providers._gguf import (
    GGUF_MAGIC,
    LARGE_ARRAY_THRESHOLD,
    ArraySummary,
    ArtifactStamp,
    GgufFormatError,
    read_gguf_header,
    sha256_of_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class TestReadingMetadata:
    def test_every_scalar_type_round_trips(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(
            tmp_path / "m.gguf",
            metadata={
                "general.architecture": "qwen3",
                "flag": True,
                "small": 40,
                "big": 1 << 40,
                "negative": -3,
                "very_negative": -(1 << 40),
                "ratio": 0.5,
                "precise": ("f64", 2.25),
            },
        )

        header = read_gguf_header(path)

        assert header.version == 3
        assert header.metadata["general.architecture"] == "qwen3"
        assert header.metadata["flag"] is True
        assert header.metadata["small"] == 40
        assert header.metadata["big"] == 1 << 40
        assert header.metadata["negative"] == -3
        assert header.metadata["very_negative"] == -(1 << 40)
        assert header.metadata["ratio"] == 0.5
        assert header.metadata["precise"] == 2.25

    def test_small_arrays_are_kept_whole_and_nested_arrays_are_read(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(
            tmp_path / "m.gguf",
            metadata={
                "heads": [8, 8, 4],
                "tags": ["a", "b"],
                "nested": [[1, 2], [3]],
                "empty": [],
            },
        )

        header = read_gguf_header(path)

        assert header.metadata["heads"] == (8, 8, 4)
        assert header.metadata["tags"] == ("a", "b")
        assert header.metadata["nested"] == ((1, 2), (3,))
        assert header.metadata["empty"] == ()

    def test_a_large_array_is_summarised_not_loaded(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        """The tokenizer vocabulary is the case: its length matters, its elements never do."""
        vocabulary = [f"token{i}" for i in range(LARGE_ARRAY_THRESHOLD + 1)]
        at_threshold = list(range(LARGE_ARRAY_THRESHOLD))
        path = gguf_writer(
            tmp_path / "m.gguf",
            metadata={"tokenizer.ggml.tokens": vocabulary, "kept": at_threshold, "after": "x"},
        )

        header = read_gguf_header(path)

        assert header.metadata["tokenizer.ggml.tokens"] == ArraySummary(
            element_type="string", length=LARGE_ARRAY_THRESHOLD + 1
        )
        assert header.metadata["kept"] == tuple(at_threshold)
        assert header.metadata["after"] == "x", "the reader must land exactly after the array"

    def test_a_large_scalar_array_is_skipped_element_by_element(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        scores = [0.5] * (LARGE_ARRAY_THRESHOLD + 3)
        path = gguf_writer(tmp_path / "m.gguf", metadata={"scores": scores, "after": "x"})

        header = read_gguf_header(path)

        assert header.metadata["scores"] == ArraySummary(element_type="float32", length=len(scores))
        assert header.metadata["after"] == "x"

    def test_a_large_array_of_arrays_is_skipped_element_by_element(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        big = [[i, i + 1] for i in range(LARGE_ARRAY_THRESHOLD + 5)]
        path = gguf_writer(tmp_path / "m.gguf", metadata={"big": big, "after": 7})

        header = read_gguf_header(path)

        assert header.metadata["big"] == ArraySummary(element_type="array", length=len(big))
        assert header.metadata["after"] == 7

    def test_parameter_count_is_the_sum_of_tensor_elements(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(
            tmp_path / "m.gguf",
            metadata={"general.architecture": "llama"},
            tensors=(("token_embd.weight", (2, 3)), ("output.bias", (4,)), ("scalar", ())),
        )

        header = read_gguf_header(path)

        assert header.tensor_count == 3
        assert header.parameter_count == 2 * 3 + 4 + 1

    def test_the_stamp_describes_the_file_that_was_read(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(tmp_path / "m.gguf", metadata={"k": 1}, payload=b"weights")

        header = read_gguf_header(path)

        assert header.path == path
        assert header.stamp == ArtifactStamp.of(path)
        assert header.stamp.size_bytes == path.stat().st_size
        assert header.stamp.key.count(":") == 3

    def test_an_unknown_key_is_kept_verbatim(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(tmp_path / "m.gguf", metadata={"mradermacher.quantized_by": "x"})

        assert read_gguf_header(path).metadata["mradermacher.quantized_by"] == "x"


class TestRefusals:
    """Every refusal is a :class:`GgufFormatError` naming the path and the reason."""

    def test_wrong_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "not.gguf"
        path.write_bytes(b"GGML" + b"\0" * 32)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert raised.value.details["path"] == str(path)
        assert "magic" in raised.value.details["reason"]
        assert isinstance(raised.value, ValidationError)
        assert raised.value.code == "GGUF_FORMAT_ERROR"

    def test_unsupported_version(self, tmp_path: Path, gguf_writer: Callable[..., Path]) -> None:
        path = gguf_writer(tmp_path / "m.gguf", metadata={"k": 1}, version=2)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "version 2" in raised.value.details["reason"]

    def test_a_file_that_ends_inside_its_header(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        whole = gguf_writer(tmp_path / "whole.gguf", metadata={"key": "a value of some length"})
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(whole.read_bytes()[:-8])

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(cut)

        assert "ended" in raised.value.details["reason"]

    def test_a_string_claiming_an_absurd_length(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1)
        key = struct.pack("<Q", 1 << 40) + b"k"
        path.write_bytes(header + key)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "cap" in raised.value.details["reason"]

    def test_a_skipped_string_claiming_an_absurd_length(self, tmp_path: Path) -> None:
        """The skip path has its own length check: a huge array of huge strings."""
        path = tmp_path / "m.gguf"
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1)
        key = struct.pack("<Q", 1) + b"k"
        array = struct.pack("<I", 9) + struct.pack("<IQ", 8, LARGE_ARRAY_THRESHOLD + 1)
        first_element = struct.pack("<Q", 1 << 40)
        path.write_bytes(header + key + array + first_element)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "cap" in raised.value.details["reason"]

    def test_an_unknown_value_type(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1)
        key = struct.pack("<Q", 1) + b"k" + struct.pack("<I", 99)
        path.write_bytes(header + key + b"\0" * 16)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "unknown value type 99" in raised.value.details["reason"]

    def test_an_unknown_array_element_type(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1)
        key = struct.pack("<Q", 1) + b"k" + struct.pack("<I", 9) + struct.pack("<IQ", 42, 1)
        path.write_bytes(header + key + b"\0" * 16)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "unknown array element type 42" in raised.value.details["reason"]

    def test_an_array_claiming_an_absurd_length(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1)
        key = struct.pack("<Q", 1) + b"k" + struct.pack("<I", 9) + struct.pack("<IQ", 4, 1 << 40)
        path.write_bytes(header + key)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "elements" in raised.value.details["reason"]

    def test_absurd_tensor_or_metadata_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(GGUF_MAGIC + struct.pack("<IQQ", 3, 1 << 40, 1))

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "sanity caps" in raised.value.details["reason"]

    def test_a_tensor_claiming_too_many_dimensions(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        header = GGUF_MAGIC + struct.pack("<IQQ", 3, 1, 0)
        tensor = struct.pack("<Q", 1) + b"t" + struct.pack("<I", 9)
        path.write_bytes(header + tensor + b"\0" * 80)

        with pytest.raises(GgufFormatError) as raised:
            read_gguf_header(path)

        assert "dimensions" in raised.value.details["reason"]

    def test_a_missing_file_raises_the_os_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_gguf_header(tmp_path / "absent.gguf")
        with pytest.raises(FileNotFoundError):
            ArtifactStamp.of(tmp_path / "absent.gguf")


class TestHashing:
    def test_the_digest_is_the_whole_file_in_normalized_form(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(tmp_path / "m.gguf", metadata={"k": 1}, payload=b"\x01\x02\x03")

        digest = sha256_of_file(path)

        assert digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert normalize_digest(digest) == digest

    def test_the_same_header_with_different_weights_hashes_differently(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        """A content digest is a digest of the content, not of the metadata."""
        first = gguf_writer(tmp_path / "a.gguf", metadata={"k": 1}, payload=b"\x01")
        second = gguf_writer(tmp_path / "b.gguf", metadata={"k": 1}, payload=b"\x02")

        assert sha256_of_file(first) != sha256_of_file(second)

    def test_a_renamed_file_keeps_its_digest(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        """Identity is the hash; the path is a locator (Adapter Identity §2)."""
        path = gguf_writer(tmp_path / "a.gguf", metadata={"k": 1}, payload=b"\x01")
        before = sha256_of_file(path)
        renamed = path.rename(tmp_path / "b.gguf")

        assert sha256_of_file(renamed) == before

    def test_the_stamp_changes_when_the_file_is_replaced(
        self, tmp_path: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        path = gguf_writer(tmp_path / "a.gguf", metadata={"k": 1}, payload=b"\x01")
        before = ArtifactStamp.of(path)
        gguf_writer(tmp_path / "a.gguf", metadata={"k": 1}, payload=b"\x01\x02")

        assert ArtifactStamp.of(path) != before
