# Local CPU workbench stack

This setup runs Qdrant together with the latest retrieval workbench UI and its
real local CPU search API. It creates:

- `aic_frames`: one point per keyframe with named `siglip2` (768) and
  `metaclip2` (1024) vectors.
- `aic_dam_regions`: one point per DAM region with a named `dam` (1024) vector
  and a `parent_point_id` back to `aic_frames`.
- `aic_beit3_frames`: one BEiT-3 COCO retrieval vector per canonical frame.

All vectors are stored as float16 in the cold memory tier. Scalar INT8 search
data is pinned in RAM, while the HNSW graph remains disk-backed. Qdrant storage
uses the local Docker named volume `aic-2026-qdrant-storage` because a Windows
bind mount does not provide the filesystem guarantees Qdrant expects.

Start Qdrant first; do not ingest until the data/model gates have passed:

```powershell
docker compose -f docker-compose.qdrant.yml up -d qdrant
```

The ingester verifies expected point IDs, named vectors, canonical payloads,
and vector direction against the offline source before considering a collection
complete. Existing collections are repaired in place in batches and every
repair is read back and verified; it never prunes unexpected points
automatically. Use `--recreate` explicitly only when the collection's immutable
vector schema is incompatible. A verified v3
`qdrant_ingestion_manifest.json` is written to the shared state volume.

Qdrant REST and dashboard are available at <http://localhost:6333> and
<http://localhost:6333/dashboard>.

The existing UI can be previewed without its legacy NumPy backend by running:

```powershell
docker compose -f docker-compose.qdrant.yml up -d ui-mock
```

Open <http://localhost:5173>. This mode validates the UI only; its API responses
are deterministic mock data and do not query Qdrant.

Prepare pinned query models, both branch data gates, and the local FTS state once,
then repair only the points reported by verification:

```powershell
docker compose -f docker-compose.qdrant.yml --profile tools run --rm query-model-setup
docker compose -f docker-compose.qdrant.yml --profile tools run --rm branch1-model-setup
docker compose -f docker-compose.qdrant.yml --profile tools run --rm branch1-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm branch2-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm text-index-prepare
docker compose -f docker-compose.qdrant.yml --profile tools run --rm qdrant-ingest --verify-only
```

`text-index-prepare` builds only the OCR database. It validates the 873 OCR
JSONL files against the 247,956 canonical keyframes, verifies FTS row/content
parity, then atomically publishes `ocr.sqlite3` (SQLite user version 3) and
`branch3_ocr_manifest.json` (`branch3.ocr-index.v3`). The API never rebuilds
OCR at startup; a missing, stale, or mixed database/manifest pair keeps the
optional OCR capability fail-closed. The v3 manifest also binds the FTS
content fingerprint and lexical-contract version to the published database.

If verification reports missing or mismatched points, repair in place and then
run the read-only command again; it must report zero repairs:

```powershell
docker compose -f docker-compose.qdrant.yml --profile tools run --rm qdrant-ingest
```

After exercising both real Branch-1 and Branch-2 searches, qualify the full
API+Qdrant Docker stack before advertising production readiness. Supply the
peak RSS measured from `docker stats` in bytes:

```powershell
python scripts/qdrant/compute_runtime_fingerprint.py --state-root <host-state-dir> --compose-file docker-compose.qdrant.yml --search-api-image-id <docker-image-id> --qdrant-image-id <docker-image-id> --query-model-manifest <host-query-model-manifest> --branch1-model-manifest <host-branch1-manifest>
$env:AIC_COMPOSE_FINGERPRINT = (Get-Content <host-state-dir>/runtime_fingerprint.json | ConvertFrom-Json).fingerprint
docker compose -f docker-compose.qdrant.yml --profile tools run --rm search-api python scripts/qdrant/qualify_resources.py --state-root /state --stack-peak-bytes <stack-bytes> --api-peak-bytes <api-bytes> --worker-peak-bytes <worker-bytes> --qdrant-peak-bytes <qdrant-bytes> --branch1-tested --branch2-tested --siglip2-tested --metaclip2-tested --bge-m3-tested --beit3-tested
```

The qualification command reads `/state/runtime_fingerprint.json` and rejects
an arbitrary fingerprint. Its digest must exactly equal the non-empty
`AIC_COMPOSE_FINGERPRINT` passed to `search-api`.

Run the real CPU-only backend and compiled UI:

```powershell
docker compose -f docker-compose.qdrant.yml up -d search-api
```

Open <http://localhost:8890>. Runtime never downloads or prewarms models.
Checkpoints are read from `aic-2026-search-model-cache` by isolated workers.
Query parsing is deterministic and local; neither Gemini nor Qwen is installed
or called. OCR and ASR use persistent SQLite FTS5 indexes in
`aic-2026-search-state`; the API never rebuilds them during startup.

The active retrieval pools are SigLIP2 full-frame cosine search, BGE-M3 DAM
region search, local OCR FTS5 keyword search, and ASR segment search using
SQLite FTS5 Okapi BM25 plus n-gram coverage. Image search, per-video search,
DAM-to-SigLIP discovery, temporal intersection, canonical submission checking,
and timeline browsing are connected to the same CPU/Qdrant backend. ASR results
are mapped back to canonical keyframes from segment metadata or timestamps.
Branch 1 uses SigLIP2, MetaCLIP2, and BEiT-3 fusion. Branch 2 uses BGE-M3 DAM
dense retrieval plus BM25 and BEiT-3 cosine validation of only the top 100.
