"""Domain-specific errors raised by VecStream."""


class VecStreamError(Exception):
    """Base class for VecStream errors."""


class InvalidVectorError(VecStreamError, ValueError):
    """A vector is malformed or contains non-finite values."""


class DimensionMismatchError(InvalidVectorError):
    """A vector does not match the configured dimensionality."""


class CorruptStoreError(VecStreamError):
    """Persisted state is incomplete, inconsistent, or corrupt."""


class InvalidManifestError(CorruptStoreError):
    """A storage manifest is missing required or consistent fields."""


class UnsupportedFormatVersionError(CorruptStoreError):
    """The store uses a format this version cannot read."""


class IndexInvariantError(VecStreamError):
    """An HNSW structural invariant was violated."""
