"""Prepare the canonical Branch-3 ASR SQLite FTS5 index.

This command is intentionally separate from OCR preparation.  It validates
all source segments, resolves their canonical frame identities, builds a
staging database, and atomically replaces only the ASR database and manifest
after every check succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online.src.retrieval.infrastructure.metadata import FrameMetadataStore  # noqa: E402
from online.src.retrieval.modalities.asr import (  # noqa: E402
    ASR_INDEX_SCHEMA_VERSION,
    AsrFtsIndex,
    artifact_record,
    build_asr_manifest,
    build_id_for,
    load_canonical_frame_index,
    validate_asr_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("AIC_DATA_ROOT", "/data"))
    )
    parser.add_argument(
        "--state-root", type=Path, default=Path(os.environ.get("AIC_STATE_ROOT", "/state"))
    )
    args = parser.parse_args()

    data_root = args.data_root
    state_root = args.state_root
    state_root.mkdir(parents=True, exist_ok=True)
    segments_dir = data_root / "asr_segments"
    database_path = state_root / "asr.sqlite3"
    manifest_path = state_root / "branch3_asr_manifest.json"
    staging_database = state_root / f".asr.sqlite3.{os.getpid()}.staging"
    staging_manifest = state_root / f".branch3_asr_manifest.{os.getpid()}.staging.json"
    canonical_path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"

    metadata = FrameMetadataStore(data_root, None)
    index: AsrFtsIndex | None = None
    try:
        staging_database.unlink(missing_ok=True)
        staging_manifest.unlink(missing_ok=True)
        source_facts = validate_asr_sources(segments_dir)
        canonical_index = load_canonical_frame_index(data_root)
        canonical_before_record = artifact_record(canonical_path)
        source_fingerprint_value = str(source_facts["source_fingerprint"])
        canonical_fingerprint_value = str(canonical_before_record["sha256"])
        build_context = {
            "source_fingerprint": source_fingerprint_value,
            "canonical_fingerprint": canonical_fingerprint_value,
            "build_id": build_id_for(
                source_fingerprint_value=source_fingerprint_value,
                canonical_fingerprint_value=canonical_fingerprint_value,
                segment_count=source_facts["segment_count"],
                video_count=source_facts["video_count"],
            ),
            "segment_count": source_facts["segment_count"],
            "video_count": source_facts["video_count"],
            "indexed_video_count": source_facts["indexed_video_count"],
            "empty_video_count": source_facts["empty_video_count"],
        }
        index = AsrFtsIndex(
            segments_dir,
            staging_database,
            metadata,
            manifest_path=staging_manifest,
            auto_prepare=True,
            canonical_frame_index=canonical_index,
            build_context=build_context,
        )
        index.validate_built_index(build_context)
        index.close()
        index = None
        # Do not publish an index assembled from a moving source tree.  The
        # source hashes were captured before mapping; size/mtime checks here
        # catch ordinary edits without another full corpus hash pass.
        for record in source_facts["source_files"]:
            source_path = data_root / str(record["path"])
            source_stat = source_path.stat()
            if int(record.get("size", -1)) != int(source_stat.st_size) or int(
                record.get("mtime_ns", -1)
            ) != int(source_stat.st_mtime_ns):
                raise RuntimeError(f"ASR source changed during preparation: {source_path}")
        manifest = build_asr_manifest(
            data_root=data_root,
            state_root=state_root,
            database_path=staging_database,
            source_facts=source_facts,
            build_context=build_context,
        )
        # The canonical map was read before the SQLite build.  Refuse to
        # publish if the source changed while mapping segments so the index
        # cannot be paired with a newer metadata file accidentally.
        recorded_canonical = manifest.get("canonical_metadata") or {}
        if (
            canonical_before_record.get("sha256") != recorded_canonical.get("sha256")
            or int(canonical_before_record.get("size", -1))
            != int(recorded_canonical.get("size", -1))
            or int(canonical_before_record.get("mtime_ns", -1))
            != int(recorded_canonical.get("mtime_ns", -1))
        ):
            raise RuntimeError("Canonical frame metadata changed during ASR preparation")
        # The stage file is moved atomically below; publish the stable public
        # database name in the manifest rather than the temporary filename.
        manifest["database"]["path"] = database_path.name
        staging_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(staging_database, database_path)
        # The manifest is written only after the database has been fully
        # committed and atomically moved into its final location.
        os.replace(staging_manifest, manifest_path)
        print(
            json.dumps(
                {
                    "schema_version": ASR_INDEX_SCHEMA_VERSION,
                    "status": "ready",
                    "passed": True,
                    "database": str(database_path),
                    "manifest": str(manifest_path),
                    "segment_count": source_facts["segment_count"],
                    "video_count": source_facts["video_count"],
                    "indexed_video_count": source_facts["indexed_video_count"],
                    "empty_video_count": source_facts["empty_video_count"],
                    "empty_video_ids": source_facts["empty_video_ids"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        if index is not None:
            index.close()
        staging_database.unlink(missing_ok=True)
        staging_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
