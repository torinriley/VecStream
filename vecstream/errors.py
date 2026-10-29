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


class WALCorruptionError(CorruptStoreError):
    """The write-ahead log is damaged somewhere other than a torn tail."""


class StoreLockedError(VecStreamError):
    """Another live process or object already owns this storage directory."""


class ClosedError(VecStreamError):
    """The collection or service was closed and no longer accepts operations."""


class AuthenticationError(VecStreamError):
    """Credentials are missing, malformed, unknown, revoked, or expired."""


class AuthorizationError(VecStreamError):
    """The authenticated principal may not perform the requested action."""
