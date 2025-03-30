# Changelog

All notable changes to VecStream will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nightly maintenance (#27)

### Fixed
- fix: create PR for changelog updates (#26)

### Fixed
- Fixed metadata filtering in HNSW search by increasing the number of candidates from 100 to 1000 to ensure enough matches are found after filtering

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
