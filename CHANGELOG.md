# Changelog

All notable changes to VecStream will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Thread-safe collections (readers-writer lock), atomic multi-operation batches, and a checksummed write-ahead log with torn-tail recovery.
- Persisted HNSW graph restored on startup instead of rebuilt; checkpoints carry SHA-256 checksums and only two generations are retained.
- Process lock per collection, collection-name validation, and fault-injection/crash tests (SIGKILL, torn WAL, bit rot).
- Versioned `/v1` HTTP API with API-key authentication, role and collection-pattern authorization, per-tenant storage isolation, Prometheus metrics, health/readiness endpoints, and graceful drain-and-checkpoint shutdown (`python -m vecstream serve`).
- Exact-search recall oracle and three reproducible ANN experiments.
- Comprehensive HNSW invariant validation and property-based mutation tests.
- Versioned, crash-safe vector checkpoints with explicit corruption errors.

### Changed
- Correct HNSW level generation, diversity selection, reciprocal pruning, and stable float32 normalization.
- Reduced the supported architecture to `Collection`, `BinaryVectorStore`, and `HNSWIndex`.

### Removed
- Legacy manager/query layers, JSON store, client/server, CLI, embedding dependencies, examples, and obsolete benchmark artifacts.

## [0.3.4] - 2024-03-30

### Fixed
- Fixed metadata filtering in HNSW search by increasing the number of candidates to ensure enough matches are found after filtering
- Ensure search_similar method properly limits results to k items
- Fixed import error by renaming VectorDBClient to ClientAPI
- Updated GitHub workflows to only test with Python 3.12

## [0.3.3] - 2024-03-21

### Added
- Initial release with core vector database functionality
- HNSW indexing for fast similarity search
- Collections/namespaces for organizing vectors
- Metadata filtering support
- Binary persistence layer
- CLI interface for basic operations 
