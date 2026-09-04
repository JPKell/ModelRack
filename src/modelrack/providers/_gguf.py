"""Domain-adjacent module — reading a GGUF file's header, and hashing its whole content.

Imports :mod:`baseaicore` and the standard library; performs **local file I/O only**, never
network I/O. This is the half of "model file management and hashing" that
[ADR-0062](../../../docs/adr/0062-llamacpp-serves-adapters-through-a-supervised-process.md)
decision 6 says the suite takes over from Ollama when it serves GGUF files itself, and it is kept
apart from :mod:`modelrack.providers.llamacpp` so that it can be tested against a file written by
a test rather than against a running server.

**Two very different costs live here, and the split between them is the point.**

* :func:`read_gguf_header` reads the *metadata* — architecture, layer count, context length,
  quantization, the tensor list — which sits at the front of the file. On the reference machine's
  5–10 GB models that is 6–16 MB, almost all of it the tokenizer vocabulary, and it costs tens of
  milliseconds. Large arrays such as the vocabulary are **skipped, not loaded**: the reader keeps
  their length and element type as an :class:`ArraySummary` and advances past the bytes.
* :func:`sha256_of_file` reads the *whole file*. That is the content digest
  [Adapter Identity §2](../../../docs/architecture/adapter-identity-and-serving.md) names as the
  artifact's identity — "identity is the hash, the path is a locator" — and it costs seconds to
  tens of seconds per file. Nothing in this module caches it; the adapter decides when the price
  is paid (see :class:`modelrack.providers.llamacpp.LlamaCppProvider`'s digest store) and this
  module deliberately offers no cheaper substitute — a header-only or size-plus-mtime hash would
  be a different identity from the one the contract names.

:class:`ArtifactStamp` is what lets the adapter *avoid* re-paying that price without changing what
the digest means: the stamp is a cheap fingerprint of the bytes' *unchangedness* (size, mtime,
inode, device), never of their content, and it decides only whether a digest already computed for
this path still describes the file.

The wire format is the one llama.cpp's ``gguf.h`` defines: ``GGUF`` magic, a version, a tensor
count, a key/value count, the key/value pairs, then one info record per tensor. All integers are
little-endian. Version 3 is what every file on the reference machine carries; version 2 differs
only in using 32-bit lengths, which this reader does not support and says so.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from baseaicore import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import BinaryIO

__all__ = [
    "GGUF_MAGIC",
    "LARGE_ARRAY_THRESHOLD",
    "ArraySummary",
    "ArtifactStamp",
    "GgufFormatError",
    "GgufHeader",
    "read_gguf_header",
    "sha256_of_file",
]

GGUF_MAGIC: Final[bytes] = b"GGUF"
"""The four bytes every GGUF file starts with."""

LARGE_ARRAY_THRESHOLD: Final[int] = 64
"""Arrays longer than this are summarised rather than kept.

A tokenizer vocabulary is 150 000–260 000 strings and the merge list is larger; nothing this
package derives from a header needs their *elements*, only their *length* (the vocabulary size).
Per-layer arrays — a head count that varies by layer, a sliding-window pattern — are a few dozen
entries and are kept whole, because :func:`modelrack.providers._llamacpp_wire.build_descriptor`
reads them.
"""

_SUPPORTED_VERSIONS: Final[frozenset[int]] = frozenset({3})
_MAX_STRING_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_ARRAY_LENGTH: Final[int] = 1 << 32
_MAX_TENSOR_DIMENSIONS: Final[int] = 8
_MAX_TENSOR_COUNT: Final[int] = 1 << 20
_MAX_KV_COUNT: Final[int] = 1 << 20

# GGUF value types, from gguf.h — the struct format for each scalar, and the two composite kinds.
_TYPE_STRING: Final[int] = 8
_TYPE_ARRAY: Final[int] = 9
_SCALAR_FORMATS: Final[dict[int, str]] = {
    0: "<B",  # UINT8
    1: "<b",  # INT8
    2: "<H",  # UINT16
    3: "<h",  # INT16
    4: "<I",  # UINT32
    5: "<i",  # INT32
    6: "<f",  # FLOAT32
    7: "<?",  # BOOL
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}
_TYPE_NAMES: Final[dict[int, str]] = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}


class GgufFormatError(ValidationError):
    """A file with a ``.gguf`` name that is not a GGUF file this reader can trust.

    Raised by :func:`read_gguf_header` for a missing magic, an unsupported version, a length
    field that cannot be honoured, or a file that ends before its own header does. A subclass of
    :class:`baseaicore.ValidationError` because that is what it is — the input failed
    validation — and so that a caller that already refuses malformed input refuses this too.
    ``details`` carries ``path`` and ``reason``.

    Never raised for a *well-formed* file whose metadata this package does not understand: an
    unknown key is kept in :attr:`GgufHeader.metadata` untouched, the same way an unknown
    provider field reaches a descriptor's ``raw``.
    """

    code = "GGUF_FORMAT_ERROR"


@dataclass(frozen=True, slots=True)
class ArraySummary:
    """What is kept of a metadata array too large to hold: its length and element type.

    Attributes:
        element_type: The GGUF type name of the elements — ``"string"``, ``"int32"``, ….
        length: How many elements the array has. For ``tokenizer.ggml.tokens`` this is the
            vocabulary size, which is the one fact about the array a descriptor needs.
    """

    element_type: str
    length: int


@dataclass(frozen=True, slots=True)
class ArtifactStamp:
    """A cheap fingerprint of whether a file's bytes have changed — never of what they are.

    Read from one :func:`os.stat` call. Two stamps that differ mean the file has certainly been
    replaced or rewritten; two that are equal mean it *almost certainly* has not — the known gap
    is a rewrite that preserves size and restores the modification time, which is deliberate
    tampering rather than any ordinary re-download, move or overwrite, and which
    ``refresh=True`` on every adapter method exists to override.

    Used by :class:`~modelrack.providers.llamacpp.LlamaCppProvider` to decide whether a content
    digest it already computed still describes the file. It is **not** an identity and never
    reaches one: the identity is the sha256 of the content, and this decides only whether that
    sha256 needs recomputing.

    Attributes:
        size_bytes: ``st_size``.
        mtime_ns: ``st_mtime_ns``.
        inode: ``st_ino``. A replaced file has a new inode even when its size and time match.
        device: ``st_dev``.
    """

    size_bytes: int
    mtime_ns: int
    inode: int
    device: int

    @classmethod
    def of(cls, path: Path) -> ArtifactStamp:
        """Read the stamp for ``path``.

        Args:
            path: The file to stat.

        Returns:
            The stamp as of now.

        Raises:
            OSError: If the file cannot be stat-ed — propagated, because a discovery that cannot
                see a file it just listed is a filesystem race worth surfacing, not hiding.
        """
        info = path.stat()
        return cls(
            size_bytes=info.st_size,
            mtime_ns=info.st_mtime_ns,
            inode=info.st_ino,
            device=info.st_dev,
        )

    @property
    def key(self) -> str:
        """Return the stamp as one string, for use inside a cache key."""
        return f"{self.size_bytes}:{self.mtime_ns}:{self.inode}:{self.device}"


@dataclass(frozen=True, slots=True)
class GgufHeader:
    """What the front of a GGUF file says about the model inside it.

    Attributes:
        path: Where it was read from.
        version: The GGUF format version (3 on every file this package has met).
        tensor_count: How many tensor records the file declares.
        metadata: Every key/value pair, keyed exactly as the file spells it
            (``general.architecture``, ``qwen3.block_count``, …). Scalars are Python scalars,
            small arrays are tuples, and arrays longer than :data:`LARGE_ARRAY_THRESHOLD` are an
            :class:`ArraySummary`. Nothing is renamed or interpreted here — that is the wire
            module's job — so this mapping is what reaches a descriptor's ``raw``.
        parameter_count: The sum of every tensor's element count. Exact, from the tensor records,
            which is why it is computed here rather than parsed from ``general.size_label``
            (``"14B"`` is a human rounding, and a rounded parameter count would corrupt the
            bytes-per-token figure FreeWeight derives from it).
        stamp: The file's :class:`ArtifactStamp` as of the read, so a cached header can be
            checked against the file it came from.
    """

    path: Path
    version: int
    tensor_count: int
    metadata: Mapping[str, Any]
    parameter_count: int
    stamp: ArtifactStamp


class _Reader:
    """A cursor over an open file that refuses to read past what the format promises."""

    __slots__ = ("_path", "_stream")

    def __init__(self, stream: BinaryIO, path: Path) -> None:
        self._stream = stream
        self._path = path

    def fail(self, reason: str) -> GgufFormatError:
        """Build the error for a structural problem at the current position."""
        return GgufFormatError(
            f"{self._path} is not a GGUF file this package can read: {reason}.",
            details={"path": str(self._path), "reason": reason},
        )

    def exact(self, count: int) -> bytes:
        """Read exactly ``count`` bytes, or raise if the file ends first."""
        data = self._stream.read(count)
        if len(data) != count:
            raise self.fail(f"file ended after {len(data)} of {count} bytes expected by the header")
        return data

    def scalar(self, type_id: int) -> Any:  # noqa: ANN401 — a GGUF scalar of any type
        """Read one scalar of the given GGUF type."""
        fmt = _SCALAR_FORMATS[type_id]
        return struct.unpack(fmt, self.exact(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        """Read one length-prefixed UTF-8 string."""
        length = self.scalar(10)
        if length > _MAX_STRING_BYTES:
            raise self.fail(f"a string claims {length} bytes, above the {_MAX_STRING_BYTES} cap")
        return self.exact(length).decode("utf-8", errors="replace")

    def value(self, type_id: int) -> Any:  # noqa: ANN401 — any GGUF value
        """Read one value of any GGUF type, summarising a large array rather than holding it."""
        if type_id == _TYPE_STRING:
            return self.string()
        if type_id == _TYPE_ARRAY:
            element_type = self.scalar(4)
            length = self.scalar(10)
            if length > _MAX_ARRAY_LENGTH:
                raise self.fail(f"an array claims {length} elements, above the cap")
            if element_type not in _TYPE_NAMES:
                raise self.fail(f"unknown array element type {element_type}")
            if length > LARGE_ARRAY_THRESHOLD:
                for _ in range(length):
                    self.skip(element_type)
                return ArraySummary(element_type=_TYPE_NAMES[element_type], length=length)
            return tuple(self.value(element_type) for _ in range(length))
        if type_id in _SCALAR_FORMATS:
            return self.scalar(type_id)
        raise self.fail(f"unknown value type {type_id}")

    def skip(self, type_id: int) -> None:
        """Advance past one value without keeping it."""
        if type_id == _TYPE_STRING:
            length = self.scalar(10)
            if length > _MAX_STRING_BYTES:
                raise self.fail(
                    f"a string claims {length} bytes, above the {_MAX_STRING_BYTES} cap"
                )
            self._stream.seek(length, 1)
        elif type_id == _TYPE_ARRAY:
            self.value(type_id)
        else:
            self._stream.seek(struct.calcsize(_SCALAR_FORMATS[type_id]), 1)


def read_gguf_header(path: Path) -> GgufHeader:
    """Read a GGUF file's metadata and tensor records, without reading its weights.

    Args:
        path: The file. Its :class:`ArtifactStamp` is read first, so the header and the stamp
            describe the same bytes unless the file changes during the read — which a later
            stamp comparison will catch.

    Returns:
        The header. Large arrays are summarised (see :data:`LARGE_ARRAY_THRESHOLD`); everything
        else is kept verbatim under the file's own key names.

    Raises:
        GgufFormatError: If the magic is wrong, the version is not one this reader supports, a
            length field exceeds a sanity cap, or the file ends inside its own header. A file
            that fails here is not a model this adapter can serve, and the adapter's discovery
            skips it with the reason recorded rather than listing a model it cannot describe.
        OSError: If the file cannot be opened or stat-ed.
    """
    stamp = ArtifactStamp.of(path)
    with path.open("rb") as stream:
        reader = _Reader(stream, path)
        if reader.exact(4) != GGUF_MAGIC:
            raise reader.fail("the file does not start with the GGUF magic bytes")
        version = reader.scalar(4)
        if version not in _SUPPORTED_VERSIONS:
            raise reader.fail(f"GGUF version {version} is not supported (supported: 3)")
        tensor_count = reader.scalar(10)
        kv_count = reader.scalar(10)
        if tensor_count > _MAX_TENSOR_COUNT or kv_count > _MAX_KV_COUNT:
            raise reader.fail(
                f"{tensor_count} tensors and {kv_count} metadata pairs exceed the sanity caps"
            )
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string()
            type_id = reader.scalar(4)
            metadata[key] = reader.value(type_id)
        parameter_count = 0
        for _ in range(tensor_count):
            reader.string()  # tensor name
            dimension_count = reader.scalar(4)
            if dimension_count > _MAX_TENSOR_DIMENSIONS:
                raise reader.fail(f"a tensor claims {dimension_count} dimensions")
            elements = 1
            for _ in range(dimension_count):
                elements *= reader.scalar(10)
            reader.scalar(4)  # ggml type
            reader.scalar(10)  # offset
            parameter_count += elements
    return GgufHeader(
        path=path,
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        parameter_count=parameter_count,
        stamp=stamp,
    )


def sha256_of_file(path: Path) -> str:
    """Return the content digest of a file, in the suite's normalized form.

    The digest [Adapter Identity §2](../../../docs/architecture/adapter-identity-and-serving.md)
    names — the sha256 of every byte of the served artifact — and the one
    :func:`baseaicore.normalize_digest` accepts without change. Reads the whole file through
    :func:`hashlib.file_digest`, so the cost is the file's size: seconds to tens of seconds for
    the multi-gigabyte files this adapter serves, which is why the adapter caches the result
    against an :class:`ArtifactStamp` rather than calling this on every discovery.

    Args:
        path: The file to hash.

    Returns:
        ``"sha256:"`` followed by 64 lowercase hex characters.

    Raises:
        OSError: If the file cannot be read.
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return f"sha256:{digest.hexdigest()}"
