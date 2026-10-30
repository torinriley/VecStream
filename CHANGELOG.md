# Changelog

All notable changes to VecStream will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Native HNSW engine in Rust (`native/`, no dependencies, C ABI via ctypes): contiguous vector slab, AVX2+FMA distance kernel with portable fallback, deletion repair, filtered traversal, GIL-free concurrent search, and multi-threaded batch search. Selected automatically when built; pure-Python engine remains the fallback. Checkpoints are engine-agnostic.
- `python -m vecstream.native build`, `serve --engine`, `--engine` for the benchmark harness, CI jobs for the Rust crate (tests, clippy) and the native test pass.
- Batch search (`Collection.search_batch`, `POST /v1/collections/{c}/search/batch`): validated all-or-nothing, one consistent snapshot, and a single matrix product on the exact path.
- Regression suite: recall floors, a distance-evaluations-per-query cost ceiling, post-deletion and filtered recall floors, and golden on-disk format tests (including v1 checkpoint compatibility).
- HNSW deletion now repairs the neighborhoods it leaves behind (recall after 20% random deletion: 0.952 vs 0.954 rebuilt, previously 0.916) and no longer scans every node to pick a new entry point.
- Filtered ANN search: metadata filters and similarity thresholds are enforced during graph traversal, with an exact-scan fallback for highly selective filters, instead of post-filtering a widened candidate list.
- `filtered` benchmark experiment; vectorized neighbor selection and pruning.
- Thread-safe collections (readers-writer lock), atomic multi-operation batches, and a checksummed write-ahead log with torn-tail recovery.
- Persisted HNSW graph restored on startup instead of rebuilt; checkpoints carry SHA-256 checksums and only two generations are retained.
- Process lock per collection, collection-name validation, and fault-injection/crash tests (SIGKILL, torn WAL, bit rot).
- Versioned `/v1` HTTP API with API-key authentication, role and collection-pattern authorization, per-tenant storage isolation, Prometheus metrics, health/readiness endpoints, and graceful drain-and-checkpoint shutdown (`python -m vecstream serve`).
- Exact-search recall oracle and three reproducible ANN experiments.
- Comprehensive HNSW invariant validation and property-based mutation tests.
- Versioned, crash-safe vector checkpoints with explicit corruption errors.

### Fixed
- The readers-writer lock is now phase-fair: a continuous stream of writes could previously starve searches indefinitely.

### Changed
- New collections default to `M=32` (was 16) with `ef_construction=200`: on GloVe-100 at 1.18M vectors this gives roughly half the search latency at equal recall, for ~20% more index memory and ~2.9x longer builds. Existing collections keep their saved parameters; omitted keys at creation are now filled from the manager defaults and saved.
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
