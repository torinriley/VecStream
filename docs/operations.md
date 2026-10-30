# Operations

## Run

```bash
python -m vecstream keys create --keys-file keys.json --tenant acme --role admin
python -m vecstream keys create --keys-file keys.json --role operator   # for metrics scraping
python -m vecstream serve --data-dir ./data --keys-file keys.json --host 0.0.0.0 --port 8080 \
    --tls-cert cert.pem --tls-key key.pem
```

The key is printed once; only a SHA-256 digest is stored, in a `0600` file that the server
re-reads on change, so `keys revoke <id>` and expiry (`--ttl-days`) apply without a restart.
Unauthenticated serving requires the explicit `--insecure-no-auth` flag (development only).
Settings may also come from `VECSTREAM_*` environment variables (`DATA_DIR`, `HOST`, `PORT`,
`KEYS_FILE`, `TLS_CERT`, `TLS_KEY`, `DRAIN_TIMEOUT`, `CHECKPOINT_OPS`, `LOG_LEVEL`).

## Engines

`python -m vecstream.native build` compiles `native/` (needs `cargo`) into `vecstream/_native/`, or set
`VECSTREAM_NATIVE_LIB` to a prebuilt library. `serve --engine auto|native|python` (`VECSTREAM_ENGINE`) picks the
index; `auto` uses native when the library loads. `python -m vecstream.native` prints availability and the active
SIMD kernel. `/v1/collections/{c}` health and `/v1/metrics` report the engine in use. Switching engines needs no
migration: the persisted graph is shared.

## Durability model

1. A write is validated, appended to `wal.log` (one checksummed record per batch, fsynced), then applied.
2. Checkpoints (every `checkpoint_ops` operations, 64 MiB of log, `POST .../checkpoint`, and shutdown) write
   `gen-*/` (vectors, ids, metadata, HNSW graph, SHA-256 manifest), atomically repoint `CURRENT`, then reset the log.
   Only the current and previous generation are kept.
3. Startup loads the newest generation whose checksums verify, restores the graph (rebuilding only if it fails
   validation), and replays log records newer than the checkpoint. The server reports `/v1/readyz` only afterwards.

## Backup and restore

`POST /v1/collections/{c}/backup` (admin) writes a verified, self-contained backup to
`<data-dir>/backups/<tenant>/<id>` while the collection keeps serving: writes and searches pause only for a
checkpoint and for hard-linking that checkpoint's immutable files; copying and verification run outside the lock.
It includes the persisted graph, so a restore does not rebuild the index. A backup is a point-in-time snapshot
taken at the checkpoint (writes acknowledged before the call are included). `POST /v1/collections/{new}/restore`
verifies every component checksum first, then creates the collection atomically; it never overwrites an existing
collection (`409`) and a damaged backup is refused with `422`. Offline equivalents, with the server stopped
(it holds the storage lock): `python -m vecstream backup --data-dir D --collection C --to PATH`,
`restore --data-dir D --collection NEW --from PATH`, and `verify-backup PATH` (any machine).

Backups live under the data directory by default; copy them to another disk or host to protect against losing
that disk. Backups are not encrypted, and a backup directory has the same sensitivity as the collection.
Restoring with a different engine works (the graph is engine-agnostic); a backup of a quantized collection needs
the native engine unless `hnsw_params` overrides `quantization`. Test a restore before you need one.

## Shutdown

`SIGTERM`/`SIGINT`: readiness fails, the listener stops, in-flight requests finish (up to `--drain-timeout`),
every collection is checkpointed, and storage locks are released. `kill -9` is also safe: the log is replayed.

## Incidents

- `CorruptStoreError ... allow_stale_checkpoint`: the newest checkpoint failed its checksum. Restore from backup, or open
  with `allow_stale_checkpoint=True` to accept losing operations since the previous checkpoint.
- `WALCorruptionError`: damage before the log tail. Restore from backup, or open with `wal_corruption="truncate"`
  to drop that record and everything after it.
- `StoreLockedError`: another process owns the directory; stop it first.
- Watch `vecstream_checkpoint_failures`, `vecstream_wal_bytes` (growing means checkpoints are failing),
  `vecstream_index_isolated_nodes`, `vecstream_http_request_duration_seconds`, and `vecstream_process_resident_memory_bytes`.

Backups: see "Backup and restore" above. There is no point-in-time recovery between backups.
