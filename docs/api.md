# HTTP API (`/v1`)

Start: `python -m vecstream serve --data-dir ./data --keys-file ./keys.json` (see
`docs/operations.md`). Every response carries `X-VecStream-API-Version: 1` and
`X-Request-ID`. Errors are `{"error": {"code": "...", "message": "..."}}`. Bodies are JSON
objects (non-finite numbers are rejected) with a required `Content-Length`.
Breaking changes will ship under a new `/v2` prefix; `/v1` only gains optional fields.

Authentication: `Authorization: Bearer vs_<id>.<secret>`. A key is bound to one tenant and a role:

| Role | Allowed |
| --- | --- |
| `reader` | list/get collections, get vectors, search |
| `writer` | reader + put/delete vectors, batches |
| `admin` | writer + create/delete/checkpoint/verify collections |
| `operator` | `/v1/metrics`, `/v1/diagnostics`, `/v1/admin/checkpoint` only; no tenant data |

Keys also carry collection name patterns (`docs-*`); access outside them is `403`.
A tenant sees only its own collections (separate storage tree), never another tenant's.

| Method & path | Role | Notes |
| --- | --- | --- |
| `GET /v1/healthz` | none | liveness |
| `GET /v1/readyz` | none | `503` while recovering or draining |
| `GET /v1/metrics` | operator | Prometheus text |
| `GET /v1/diagnostics` | operator | memory, latency bounds, per-collection health |
| `POST /v1/admin/checkpoint` | operator | checkpoint every open collection |
| `GET /v1/collections` | reader | names visible to the key |
| `POST /v1/collections` | admin | `{"name", "use_hnsw"?, "hnsw_params"?: {"M","ef_construction","ml","quantization"}}` → `201` |
| `GET /v1/collections/{c}` | reader | stats and index/WAL health |
| `DELETE /v1/collections/{c}` | admin | |
| `POST /v1/collections/{c}/checkpoint` · `/verify` | admin | verify re-checks checksums and graph invariants |
| `POST /v1/collections/{c}/backup` | admin | `{"id"?}` → `201`; writes a verified backup under `<data-dir>/backups/<tenant>/<id>` (default id `<c>-<UTC timestamp>`); `409` if the id exists |
| `GET /v1/backups` | admin | this tenant's backups for collections the key may manage |
| `POST /v1/collections/{c}/restore` | admin | `{"backup": id, "hnsw_params"?}` → `201`; creates collection `c` from the backup (`409` if `c` exists, `422` if the backup fails verification); the key must be allowed to manage the backup's source collection too |
| `POST /v1/collections/{c}/search/batch` | reader | `{"vectors": [[...], ...], "k"?, "threshold"?, "ef_search"?, "filter"?}` → `{"results": [[...], ...]}`; ≤ 1,000 queries, one consistent snapshot |
| `PUT /v1/collections/{c}/vectors/{id}` | writer | `{"vector": [...], "metadata"?: {...}}` upsert (replaces metadata) |
| `GET` / `DELETE /v1/collections/{c}/vectors/{id}` | reader / writer | |
| `POST /v1/collections/{c}/batch` | writer | `{"operations": [{"op":"put"\|"del","id",...}]}` — atomic, ≤ 10,000 ops |
| `POST /v1/collections/{c}/search` | reader | `{"vector","k"?,"threshold"?,"ef_search"?,"filter"?,"include_metadata"?}` |

A `2xx` write response means the operation is in the fsynced write-ahead log.
