"""Colab A100 pipeline for downloading and embedding AIC keyframes.

The two Colab notebooks in ``notebooks/`` call this module with either the
MetaCLIP 2 worldwide encoder or the official BEiT-3 COCO retrieval encoder.
Drive files are downloaded one ZIP at a time, safely extracted, and removed so
the 31 GB compressed corpus does not occupy disk alongside the extracted data.
Embeddings are committed per video before a final, deterministic merge.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
import tempfile
import time
import types
import zipfile
from collections import Counter, abc, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_FOLDER_IDS = (
    "1ZjLlGH0Igq70wrAELVIU4kLIWSFUAN4B",
    "1nxum5Qp5_iCQIqud11I8u8ACKE8OOsv5",
)
DEFAULT_DATA_ROOT_IDS = (
    "1ZZpoqN-gehvKO9jUG45Gr3wFIcIcbAta",
    "1_8Y2PWseN5Max2Lwi43_DW_ulo-T-NzE",
)
DEFAULT_MAP_FOLDER_IDS = (
    "1t-BCtvBPpvG4T0-wKNWFmuHEA0_uXOzN",
    "1xdP7xe9KfneAAlSo_yCPXgE8u0jksvSs",
)
DEFAULT_METACLIP2_ID = "facebook/metaclip-2-worldwide-huge-quickgelu"
DEFAULT_BEIT3_CHECKPOINT = (
    "https://github.com/addf400/files/releases/download/beit3/"
    "beit3_base_patch16_384_coco_retrieval.pth"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_ID_RE = re.compile(r"L\d+_V\d+", re.IGNORECASE)


def natural_key(value: str) -> tuple[Any, ...]:
    """Return a stable human/numeric ordering key."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def list_drive_archives(service: Any, folder_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Page through all ZIP children of the supplied shared Drive folders."""
    archives: list[dict[str, Any]] = []
    for folder_id in folder_ids:
        page_token: str | None = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken,files(id,name,size,mimeType,md5Checksum)",
                    pageSize=1000,
                    pageToken=page_token,
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute(num_retries=5)
            )
            for item in response.get("files", []):
                if item.get("name", "").lower().endswith(".zip"):
                    item["source_folder_id"] = folder_id
                    archives.append(item)
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    ids = [item["id"] for item in archives]
    names = [item["name"] for item in archives]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Drive listing contains duplicate file IDs")
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1}, key=natural_key)
        raise RuntimeError(f"Duplicate ZIP names across Drive folders: {duplicates[:10]}")
    return sorted(archives, key=lambda item: natural_key(item["name"]))


def resolve_data_roots(
    service: Any,
    root_folder_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Resolve keyframes_zips and map-keyframes children from each data root."""
    keyframe_folder_ids: list[str] = []
    map_folder_ids: list[str] = []
    for root_id in root_folder_ids:
        response = (
            service.files()
            .list(
                q=f"'{root_id}' in parents and trashed = false",
                fields="files(id,name,mimeType)",
                pageSize=1000,
                spaces="drive",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute(num_retries=5)
        )
        children = {
            item["name"]: item
            for item in response.get("files", [])
            if item.get("mimeType") == "application/vnd.google-apps.folder"
        }
        missing = {"keyframes_zips", "map-keyframes"} - set(children)
        if missing:
            raise RuntimeError(f"Data root {root_id} is missing folders: {sorted(missing)}")
        keyframe_folder_ids.append(children["keyframes_zips"]["id"])
        map_folder_ids.append(children["map-keyframes"]["id"])
    return keyframe_folder_ids, map_folder_ids


def load_drive_keyframe_maps(
    service: Any,
    folder_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load and validate all organizer map CSVs directly from shared Drive."""
    from tqdm.auto import tqdm

    files: list[dict[str, Any]] = []
    for folder_id in folder_ids:
        page_token: str | None = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken,files(id,name,size,mimeType,md5Checksum)",
                    pageSize=1000,
                    pageToken=page_token,
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute(num_retries=5)
            )
            for item in response.get("files", []):
                if item.get("name", "").lower().endswith(".csv"):
                    item["source_folder_id"] = folder_id
                    files.append(item)
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    maps: dict[str, list[dict[str, Any]]] = {}
    ordered_files = sorted(files, key=lambda value: natural_key(value["name"]))
    print(f"Reading {len(ordered_files)} map-keyframes CSVs from shared Drive")
    for item in tqdm(ordered_files, desc="Drive keyframe maps", unit="csv"):
        video_id = Path(item["name"]).stem.upper()
        if video_id in maps:
            raise RuntimeError(f"Duplicate map CSV for {video_id}")
        payload = (
            service.files()
            .get_media(fileId=item["id"], supportsAllDrives=True)
            .execute(num_retries=5)
        )
        text = payload.decode("utf-8-sig")
        rows = []
        for row in csv.DictReader(io.StringIO(text)):
            rows.append(
                {
                    "keyframe_n": int(row["n"]),
                    "pts_time_s": float(row["pts_time"]),
                    "fps": float(row["fps"]),
                    "frame_idx": int(row["frame_idx"]),
                }
            )
        numbers = [row["keyframe_n"] for row in rows]
        frame_indices = [row["frame_idx"] for row in rows]
        timestamps = [row["pts_time_s"] for row in rows]
        if not rows or numbers != sorted(set(numbers)):
            raise RuntimeError(f"Invalid keyframe sequence in {item['name']}")
        if frame_indices != sorted(frame_indices) or timestamps != sorted(timestamps):
            raise RuntimeError(f"Non-monotonic map rows in {item['name']}")
        if any(row["fps"] <= 0 for row in rows):
            raise RuntimeError(f"Non-positive FPS in {item['name']}")
        maps[video_id] = rows
    return maps


def load_cached_keyframe_maps(
    service: Any,
    folder_ids: Sequence[str],
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Cache the small organizer CSV snapshot on persistent output storage."""
    cache_path = output_dir / "keyframe_maps_snapshot.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("source_map_folder_ids") == list(folder_ids):
            maps = payload.get("maps")
            if isinstance(maps, dict) and maps:
                print(f"Using cached keyframe maps: {cache_path}")
                return {video_id.upper(): rows for video_id, rows in maps.items()}

    maps = load_drive_keyframe_maps(service, folder_ids)
    atomic_json(
        cache_path,
        {
            "source_map_folder_ids": list(folder_ids),
            "video_count": len(maps),
            "maps": maps,
        },
    )
    return maps


def validate_archive_map_pairs(
    archives: Sequence[dict[str, Any]],
    keyframe_maps: dict[str, list[dict[str, Any]]],
) -> None:
    archive_ids = {Path(item["name"]).stem.upper() for item in archives}
    map_ids = set(keyframe_maps)
    if archive_ids != map_ids:
        missing_maps = sorted(archive_ids - map_ids, key=natural_key)
        missing_archives = sorted(map_ids - archive_ids, key=natural_key)
        raise RuntimeError(
            "Drive keyframes/map-keyframes pairing failed: "
            f"ZIPs without CSV={missing_maps[:10]}, CSVs without ZIP={missing_archives[:10]}"
        )
    print(f"Verified exact Drive pairing: {len(archive_ids)} ZIPs + {len(map_ids)} map CSVs")


def ensure_embedding_identity(path: Path, identity: dict[str, Any]) -> None:
    """Lock output to one model/corpus while accepting the legacy child-folder identity."""
    if not path.exists():
        atomic_json(path, identity)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    scalar_fields = ("model_family", "model_id", "dtype", "l2_normalized")
    compatible = all(existing.get(field) == identity.get(field) for field in scalar_fields)
    compatible = compatible and set(existing.get("source_drive_folder_ids", [])) == set(
        identity["source_drive_folder_ids"]
    )
    for field in ("source_data_root_folder_ids", "source_map_folder_ids"):
        if field in existing:
            compatible = compatible and set(existing[field]) == set(identity[field])
    if not compatible:
        raise RuntimeError(
            f"Output directory belongs to a different embedding run: {existing}. "
            "Choose a new --output-dir."
        )
    if existing != identity:
        atomic_json(path, identity)
        print("Upgraded legacy embedding identity with data-root/map folder IDs")


def safe_extract(archive_path: Path, destination: Path) -> int:
    """Extract a ZIP while rejecting absolute paths and path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted_files = 0
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member {bad!r} in {archive_path.name}")
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe ZIP member {member.filename!r}") from exc
            archive.extract(member, destination)
            if not member.is_dir():
                extracted_files += 1
    return extracted_files


def download_and_extract_archives(
    service: Any,
    archives: Sequence[dict[str, Any]],
    data_dir: Path,
    zip_cache_dir: Path,
    keep_zips: bool = False,
) -> None:
    """Download each Drive archive, verify its size, extract it, and checkpoint."""
    from googleapiclient.http import MediaIoBaseDownload
    from tqdm.auto import tqdm

    marker_dir = data_dir / ".drive_extract_complete"
    marker_dir.mkdir(parents=True, exist_ok=True)
    zip_cache_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(archives, desc="Drive ZIPs", unit="zip"):
        marker = marker_dir / f"{item['id']}.json"
        if marker.exists():
            continue

        zip_path = zip_cache_dir / item["name"]
        partial_path = zip_path.with_suffix(zip_path.suffix + ".part")
        expected_size = int(item.get("size") or 0)
        if not zip_path.exists() or (expected_size and zip_path.stat().st_size != expected_size):
            partial_path.unlink(missing_ok=True)
            request = service.files().get_media(fileId=item["id"], supportsAllDrives=True)
            with partial_path.open("wb") as handle:
                downloader = MediaIoBaseDownload(handle, request, chunksize=64 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=5)
            if expected_size and partial_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {item['name']}: "
                    f"{partial_path.stat().st_size} != {expected_size}"
                )
            partial_path.replace(zip_path)

        # Always isolate one archive per video.  This supports both ZIP layouts:
        # images directly at archive root and images under an Lxx_Vxxx folder.
        archive_destination = data_dir / Path(item["name"]).stem
        extracted_files = safe_extract(zip_path, archive_destination)
        atomic_json(
            marker,
            {
                "drive_file_id": item["id"],
                "name": item["name"],
                "size": expected_size,
                "md5_checksum": item.get("md5Checksum"),
                "extracted_files": extracted_files,
                "destination": archive_destination.relative_to(data_dir).as_posix(),
            },
        )
        if not keep_zips:
            zip_path.unlink(missing_ok=True)


def infer_video_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = VIDEO_ID_RE.fullmatch(part)
        if match:
            return match.group(0).upper()
    match = VIDEO_ID_RE.search(path.as_posix())
    if match:
        return match.group(0).upper()
    raise ValueError(f"Cannot infer video ID from {path}")


def discover_keyframes(data_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            grouped[infer_video_id(path)].append(path)
    return {
        video_id: sorted(paths, key=lambda path: natural_key(path.as_posix()))
        for video_id, paths in sorted(grouped.items(), key=lambda pair: natural_key(pair[0]))
    }


def keyframe_number(path: Path, fallback: int) -> int:
    del fallback
    if not path.stem.isdigit():
        raise ValueError(
            f"Keyframe filename must be numeric (for example 001.jpg), got {path.name!r}"
        )
    return int(path.stem)


def collate_image_batch(batch: Sequence[tuple[str, Any]]) -> tuple[list[str], list[Any]]:
    paths, images = zip(*batch)
    return list(paths), list(images)


class KeyframeDataset:
    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, Any]:
        from PIL import Image, ImageOps

        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.load()
                return str(path), image
        except Exception as exc:
            raise RuntimeError(f"Cannot decode keyframe {path}") from exc


def unwrap_pooler_output(value: Any) -> Any:
    """Support both older and newer Transformers feature return types."""
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    if hasattr(value, "image_embeds"):
        return value.image_embeds
    return value


class MetaClip2Embedder:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    @property
    def preprocessing(self) -> dict[str, Any]:
        return {"source": "Hugging Face AutoProcessor", "model_id": self.model_id}

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        if self.device == "cuda":
            pixel_values = pixel_values.half()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.device == "cuda",
        ):
            outputs = self.model.get_image_features(pixel_values=pixel_values)
            features = unwrap_pooler_output(outputs)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        return features.cpu().numpy().astype(np.float16, copy=False)


def download_http_resumable(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60,
        allow_redirects=True,
    ) as response:
        if existing and response.status_code == 200:
            partial.unlink(missing_ok=True)
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


class BEiT3Embedder:
    def __init__(self, repo_dir: Path, checkpoint_url: str, checkpoint_dir: Path, device: str):
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        beit3_dir = repo_dir / "beit3" if (repo_dir / "beit3").is_dir() else repo_dir
        if not (beit3_dir / "modeling_finetune.py").exists():
            raise FileNotFoundError(f"BEiT-3 source not found under {beit3_dir}")

        # timm 0.4.x and the original training utilities predate torch 2.x.
        # These compatibility symbols were containers/constants, not tensor
        # implementations, so a narrow shim is sufficient for inference.
        torch_six_shim = types.ModuleType("torch._six")
        torch_six_shim.container_abcs = abc
        torch_six_shim.inf = float("inf")
        torch_six_shim.string_classes = (str,)
        sys.modules.setdefault("torch._six", torch_six_shim)

        # modeling_finetune only needs these symbols for inference.  The shim
        # avoids importing unrelated legacy training dependencies.
        utils_shim = types.ModuleType("utils")
        utils_shim.get_rank = lambda: 0
        utils_shim.get_world_size = lambda: 1

        class ClipLoss(nn.Module):
            def forward(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(
                    "ClipLoss is training-only and unavailable in this inference notebook"
                )

        utils_shim.ClipLoss = ClipLoss
        sys.modules["utils"] = utils_shim
        sys.path.insert(0, str(beit3_dir))
        from modeling_finetune import beit3_base_patch16_384_retrieval

        checkpoint_path = checkpoint_dir / Path(checkpoint_url).name
        if not checkpoint_path.exists():
            download_http_resumable(checkpoint_url, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model = beit3_base_patch16_384_retrieval()
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"BEiT-3 checkpoint mismatch: {incompatible}")

        self.torch = torch
        self.device = device
        self.checkpoint_url = checkpoint_url
        self.model = model.to(device).eval()
        if device == "cuda":
            self.model.half()
        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @property
    def preprocessing(self) -> dict[str, Any]:
        return {
            "resize": [384, 384],
            "interpolation": "bicubic",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "checkpoint": self.checkpoint_url,
        }

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        torch = self.torch
        batch = torch.stack([self.transform(image) for image in images]).to(
            self.device, non_blocking=True
        )
        if self.device == "cuda":
            batch = batch.half()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.device == "cuda",
        ):
            vision_features, _ = self.model(image=batch, only_infer=True)
            vision_features = torch.nn.functional.normalize(vision_features.float(), p=2, dim=-1)
        return vision_features.cpu().numpy().astype(np.float16, copy=False)


def create_embedder(args: argparse.Namespace, device: str) -> Any:
    if args.model == "metaclip2":
        return MetaClip2Embedder(args.metaclip2_model_id, device)
    return BEiT3Embedder(
        Path(args.beit3_repo_dir),
        args.beit3_checkpoint_url,
        Path(args.checkpoint_dir),
        device,
    )


def save_array_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def save_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_video_metadata(
    video_id: str,
    paths: Sequence[Path],
    map_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    map_by_number = {int(row["keyframe_n"]): row for row in map_rows}
    image_numbers = [keyframe_number(path, index) for index, path in enumerate(paths, start=1)]
    map_numbers = [int(row["keyframe_n"]) for row in map_rows]
    if image_numbers != map_numbers:
        raise RuntimeError(
            f"Image/map mismatch for {video_id}: {len(image_numbers)} images versus "
            f"{len(map_numbers)} map rows; image sample={image_numbers[:5]}, "
            f"map sample={map_numbers[:5]}"
        )

    records = []
    for vector_row, (path, number) in enumerate(zip(paths, image_numbers, strict=True)):
        mapping = map_by_number[number]
        frame_idx = int(mapping["frame_idx"])
        records.append(
            {
                "video_id": video_id,
                "keyframe_n": number,
                "frame_idx": frame_idx,
                "pts_time_s": float(mapping["pts_time_s"]),
                "fps": float(mapping["fps"]),
                "frame_uid": f"{video_id}:{frame_idx}",
                "vector_row": vector_row,
                "vector_shard": f"shards/{video_id}.f16.npy",
                "image_relpath": f"keyframes/{video_id}/{path.name}",
                "filename": path.name,
            }
        )
    numbers = [record["keyframe_n"] for record in records]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError(f"Duplicate keyframe_n values for {video_id}")
    if numbers != sorted(numbers):
        raise RuntimeError(f"keyframe_n values are not increasing for {video_id}")
    return records


def completed_stream_shard(
    output_dir: Path,
    video_id: str,
    map_rows: Sequence[dict[str, Any]],
) -> tuple[int, int] | None:
    shard_path = output_dir / "shards" / f"{video_id}.f16.npy"
    metadata_path = output_dir / "video_metadata" / f"{video_id}.jsonl"
    if not shard_path.exists() or not metadata_path.exists():
        return None
    shard = np.load(shard_path, mmap_mode="r")
    with metadata_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    metadata_rows = len(records)
    valid = (
        shard.ndim == 2
        and shard.dtype == np.float16
        and shard.shape[0] == metadata_rows
        and metadata_rows > 0
    )
    if not valid:
        raise RuntimeError(f"Invalid existing streaming shard for {video_id}")
    map_by_number = {int(row["keyframe_n"]): row for row in map_rows}
    record_numbers = [int(record.get("keyframe_n", -1)) for record in records]
    map_numbers = [int(row["keyframe_n"]) for row in map_rows]
    if record_numbers != map_numbers:
        raise RuntimeError(
            f"Existing shard/map mismatch for {video_id}: "
            f"{len(record_numbers)} vector rows versus {len(map_numbers)} map rows"
        )

    migrated = False
    for vector_row, record in enumerate(records):
        if record.get("video_id") != video_id or "keyframe_n" not in record:
            raise RuntimeError(f"Invalid mapping record for {video_id} row {vector_row}")
        mapping = map_by_number[int(record["keyframe_n"])]
        expected_mapping = {
            "frame_idx": int(mapping["frame_idx"]),
            "pts_time_s": float(mapping["pts_time_s"]),
            "fps": float(mapping["fps"]),
            "frame_uid": f"{video_id}:{int(mapping['frame_idx'])}",
        }
        expected_shard = f"shards/{video_id}.f16.npy"
        if record.get("vector_row") != vector_row:
            record["vector_row"] = vector_row
            migrated = True
        if record.get("vector_shard") != expected_shard:
            record["vector_shard"] = expected_shard
            migrated = True
        for field, expected_value in expected_mapping.items():
            if record.get(field) != expected_value:
                record[field] = expected_value
                migrated = True
    if migrated:
        save_jsonl_atomic(metadata_path, records)
    return int(shard.shape[1]), metadata_rows


def stream_drive_archives(
    service: Any,
    archives: Sequence[dict[str, Any]],
    keyframe_maps: dict[str, list[dict[str, Any]]],
    embedder: Any,
    output_dir: Path,
    work_dir: Path,
    zip_cache_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[list[str], int, int]:
    """Download, embed, and discard one shared-Drive ZIP at a time."""
    from tqdm.auto import tqdm

    work_dir.mkdir(parents=True, exist_ok=True)
    video_ids: list[str] = []
    embedding_dim = 0
    total_images = 0

    for item in tqdm(archives, desc="Streaming Drive videos", unit="video"):
        video_id = Path(item["name"]).stem.upper()
        if not VIDEO_ID_RE.fullmatch(video_id):
            raise RuntimeError(f"Unexpected Drive ZIP name: {item['name']}")
        if video_id not in keyframe_maps:
            raise RuntimeError(f"No map-keyframes CSV found for {video_id}")
        video_ids.append(video_id)

        completed = completed_stream_shard(output_dir, video_id, keyframe_maps[video_id])
        if completed is not None:
            shard_dim, row_count = completed
            if embedding_dim and shard_dim != embedding_dim:
                raise RuntimeError(f"Embedding dimension mismatch in shard {video_id}")
            embedding_dim = shard_dim
            total_images += row_count
            continue

        with tempfile.TemporaryDirectory(prefix=f"{video_id}-", dir=work_dir) as temporary:
            temporary_dir = Path(temporary)
            download_and_extract_archives(
                service,
                [item],
                temporary_dir,
                zip_cache_dir,
                keep_zips=False,
            )
            grouped = discover_keyframes(temporary_dir)
            paths = grouped.get(video_id, [])
            if not paths:
                raise RuntimeError(f"ZIP {item['name']} contains no images for {video_id}")
            unexpected = sorted(set(grouped) - {video_id}, key=natural_key)
            if unexpected:
                raise RuntimeError(
                    f"ZIP {item['name']} unexpectedly contains other videos: {unexpected}"
                )

            shard_dim = embed_video_shards(
                {video_id: paths},
                embedder,
                output_dir,
                batch_size,
                num_workers,
                device,
            )
            records = build_video_metadata(video_id, paths, keyframe_maps[video_id])
            save_jsonl_atomic(
                output_dir / "video_metadata" / f"{video_id}.jsonl",
                records,
            )
            if embedding_dim and shard_dim != embedding_dim:
                raise RuntimeError(f"Embedding dimension changed at {video_id}")
            embedding_dim = shard_dim
            total_images += len(records)

    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("Duplicate video IDs in Drive archive manifest")
    return video_ids, embedding_dim, total_images


def merge_stream_outputs(
    video_ids: Sequence[str],
    output_dir: Path,
    embedding_dim: int,
) -> tuple[Path, Path, int]:
    shard_rows: list[tuple[str, int]] = []
    for video_id in video_ids:
        shard = np.load(output_dir / "shards" / f"{video_id}.f16.npy", mmap_mode="r")
        if shard.ndim != 2 or shard.shape[1] != embedding_dim:
            raise RuntimeError(f"Invalid shard shape for {video_id}: {shard.shape}")
        shard_rows.append((video_id, int(shard.shape[0])))

    total = sum(rows for _, rows in shard_rows)
    matrix_path = output_dir / "keyframes_visual_vectors.f16.npy"
    metadata_path = output_dir / "keyframes_metadata.jsonl"
    temporary_matrix = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary_matrix,
        mode="w+",
        dtype=np.float16,
        shape=(total, embedding_dim),
    )
    offset = 0
    with temporary_metadata.open("w", encoding="utf-8") as destination:
        for video_id, rows in shard_rows:
            shard = np.load(output_dir / "shards" / f"{video_id}.f16.npy", mmap_mode="r")
            matrix[offset : offset + rows] = shard
            source_path = output_dir / "video_metadata" / f"{video_id}.jsonl"
            with source_path.open("r", encoding="utf-8") as source:
                for local_index, line in enumerate(source):
                    record = json.loads(line)
                    global_row = offset + local_index
                    record["row_id"] = global_row
                    record["global_vector_row"] = global_row
                    record["point_id"] = global_row + 1
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            offset += rows
    matrix.flush()
    del matrix
    temporary_matrix.replace(matrix_path)
    temporary_metadata.replace(metadata_path)
    return matrix_path, metadata_path, total


def export_keyframe_index_csv(metadata_path: Path, output_dir: Path) -> Path:
    """Export a human-readable row-to-keyframe mapping beside the NPY matrix."""
    index_path = output_dir / "keyframe_index.csv"
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    fieldnames = [
        "global_vector_row",
        "point_id",
        "video_id",
        "keyframe_n",
        "frame_idx",
        "pts_time_s",
        "fps",
        "frame_uid",
        "vector_shard",
        "vector_row",
        "filename",
        "image_relpath",
    ]
    with (
        metadata_path.open("r", encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8", newline="") as destination,
    ):
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for line in source:
            record = json.loads(line)
            writer.writerow({field: record.get(field) for field in fieldnames})
    temporary.replace(index_path)
    return index_path


def embed_video_shards(
    grouped_paths: dict[str, list[Path]],
    embedder: Any,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> int:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    embedding_dim = 0
    for video_id, paths in tqdm(grouped_paths.items(), desc="Videos", unit="video"):
        shard_path = shard_dir / f"{video_id}.f16.npy"
        if shard_path.exists():
            existing = np.load(shard_path, mmap_mode="r")
            valid_existing = (
                existing.ndim == 2
                and existing.shape[0] == len(paths)
                and existing.dtype == np.float16
            )
            if valid_existing:
                embedding_dim = int(existing.shape[1])
                continue
            raise RuntimeError(f"Invalid existing shard; remove or repair it: {shard_path}")

        loader = DataLoader(
            KeyframeDataset(paths),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_image_batch,
            persistent_workers=num_workers > 0,
        )
        batches: list[np.ndarray] = []
        observed_paths: list[str] = []
        for batch_paths, images in loader:
            batches.append(embedder.encode(images))
            observed_paths.extend(batch_paths)
        if observed_paths != [str(path) for path in paths]:
            raise RuntimeError(f"DataLoader changed keyframe order for {video_id}")
        vectors = np.concatenate(batches, axis=0)
        if vectors.shape[0] != len(paths) or not np.isfinite(vectors).all():
            raise RuntimeError(f"Invalid vectors generated for {video_id}: {vectors.shape}")
        norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3):
            raise RuntimeError(f"Non-normalized vectors generated for {video_id}")
        embedding_dim = int(vectors.shape[1])
        save_array_atomic(shard_path, vectors)
        del vectors, batches
        if device == "cuda":
            torch.cuda.empty_cache()
    return embedding_dim


def merge_outputs(
    grouped_paths: dict[str, list[Path]],
    output_dir: Path,
    data_dir: Path,
    embedding_dim: int,
) -> tuple[Path, Path, int]:
    total = sum(len(paths) for paths in grouped_paths.values())
    matrix_path = output_dir / "keyframes_visual_vectors.f16.npy"
    metadata_path = output_dir / "keyframes_metadata.jsonl"
    temporary_matrix = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary_matrix,
        mode="w+",
        dtype=np.float16,
        shape=(total, embedding_dim),
    )
    offset = 0
    with temporary_metadata.open("w", encoding="utf-8") as metadata_file:
        for video_id, paths in grouped_paths.items():
            shard = np.load(output_dir / "shards" / f"{video_id}.f16.npy", mmap_mode="r")
            matrix[offset : offset + len(paths)] = shard
            for local_index, path in enumerate(paths, start=1):
                record = {
                    "row_id": offset + local_index - 1,
                    "global_vector_row": offset + local_index - 1,
                    "video_id": video_id,
                    "keyframe_n": keyframe_number(path, local_index),
                    "vector_row": local_index - 1,
                    "vector_shard": f"shards/{video_id}.f16.npy",
                    "image_relpath": path.relative_to(data_dir).as_posix(),
                    "filename": path.name,
                }
                metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            offset += len(paths)
    matrix.flush()
    del matrix
    temporary_matrix.replace(matrix_path)
    temporary_metadata.replace(metadata_path)
    return matrix_path, metadata_path, total


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("metaclip2", "beit3"), required=True)
    parser.add_argument("--folder-id", action="append", dest="folder_ids")
    parser.add_argument(
        "--data-root-folder-id",
        action="append",
        dest="data_root_folder_ids",
        help="Shared Drive data root containing keyframes_zips and map-keyframes.",
    )
    parser.add_argument("--map-folder-id", action="append", dest="map_folder_ids")
    parser.add_argument("--data-dir", default="/content/aic_keyframes")
    parser.add_argument("--zip-cache-dir", default="/content/aic_zip_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="/content/model_checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-zips", type=int)
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--stream-archives",
        action="store_true",
        help="Embed one shared-Drive ZIP at a time and discard its temporary images.",
    )
    parser.add_argument("--expected-videos", type=int, default=873)
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Optional exact image count. Omit when the Drive corpus can grow.",
    )
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--metaclip2-model-id", default=DEFAULT_METACLIP2_ID)
    parser.add_argument("--beit3-repo-dir", default="/content/unilm")
    parser.add_argument("--beit3-checkpoint-url", default=DEFAULT_BEIT3_CHECKPOINT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_root_folder_ids = tuple(args.data_root_folder_ids or ())
    folder_ids = tuple(args.folder_ids or ())
    map_folder_ids = tuple(args.map_folder_ids or ())
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stream_archives and args.skip_download:
        raise ValueError("--stream-archives and --skip-download cannot be used together")

    if not args.skip_download:
        import google.auth
        import google_auth_httplib2
        import httplib2
        from googleapiclient.discovery import build

        credentials, _ = google.auth.default()
        authorized_http = google_auth_httplib2.AuthorizedHttp(
            credentials,
            http=httplib2.Http(timeout=300),
        )
        # New google-auth versions pass a redundant per-request refresh timeout.
        # httplib2 ignores it and logs a warning even though the socket-level
        # timeout above is active, so silence only that transport logger.
        logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)
        service = build(
            "drive",
            "v3",
            http=authorized_http,
            cache_discovery=False,
        )
        if data_root_folder_ids:
            if folder_ids or map_folder_ids:
                raise ValueError(
                    "Use --data-root-folder-id by itself; child keyframe/map folders "
                    "are discovered automatically."
                )
            resolved_keyframes, resolved_maps = resolve_data_roots(
                service, data_root_folder_ids
            )
            folder_ids = tuple(resolved_keyframes)
            map_folder_ids = tuple(resolved_maps)
        else:
            folder_ids = folder_ids or DEFAULT_FOLDER_IDS
            map_folder_ids = map_folder_ids or DEFAULT_MAP_FOLDER_IDS

        all_archives = list_drive_archives(service, folder_ids)
        keyframe_maps = load_cached_keyframe_maps(service, map_folder_ids, output_dir)
        validate_archive_map_pairs(all_archives, keyframe_maps)
        archives = all_archives
        if args.max_zips is not None:
            archives = archives[: args.max_zips]
        archive_counts = Counter(item["source_folder_id"] for item in archives)
        print("Drive folders selected:")
        for folder_id in folder_ids:
            print(f"  - {folder_id}: {archive_counts[folder_id]} ZIPs")
        compressed_gib = sum(int(item.get("size") or 0) for item in archives) / 1024**3
        print(f"Drive manifest: {len(archives)} ZIPs, {compressed_gib:.2f} GiB compressed")
        atomic_json(output_dir / "drive_archives_manifest.json", archives)
        if args.stream_archives:
            identity = {
                "model_family": args.model,
                "model_id": (
                    args.metaclip2_model_id
                    if args.model == "metaclip2"
                    else args.beit3_checkpoint_url
                ),
                "source_drive_folder_ids": list(folder_ids),
                "source_data_root_folder_ids": list(data_root_folder_ids),
                "source_map_folder_ids": list(map_folder_ids),
                "dtype": "float16",
                "l2_normalized": True,
            }
            identity_path = output_dir / "embedding_identity.json"
            ensure_embedding_identity(identity_path, identity)

            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required; choose an A100 runtime in Colab")
            device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(
                f"GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)"
            )

            started = time.time()
            embedder = create_embedder(args, device)
            video_ids, embedding_dim, image_count = stream_drive_archives(
                service,
                archives,
                keyframe_maps,
                embedder,
                output_dir,
                data_dir,
                Path(args.zip_cache_dir),
                args.batch_size,
                args.num_workers,
                device,
            )
            video_count = len(video_ids)
            video_count_mismatch = video_count != args.expected_videos
            image_count_mismatch = (
                args.expected_images is not None and image_count != args.expected_images
            )
            count_mismatch = video_count_mismatch or image_count_mismatch
            if not args.allow_count_mismatch and args.max_zips is None and count_mismatch:
                expected_images = (
                    args.expected_images if args.expected_images is not None else "dynamic"
                )
                raise RuntimeError(
                    "Corpus count mismatch: "
                    f"got {video_count} videos/{image_count} images; expected "
                    f"{args.expected_videos}/{expected_images}."
                )

            matrix_path, metadata_path, total = merge_stream_outputs(
                video_ids,
                output_dir,
                embedding_dim,
            )
            matrix = np.load(matrix_path, mmap_mode="r")
            metadata_lines = sum(1 for _ in metadata_path.open("r", encoding="utf-8"))
            if matrix.shape != (total, embedding_dim) or metadata_lines != total:
                raise RuntimeError("Final streaming matrix/metadata verification failed")
            index_path = export_keyframe_index_csv(metadata_path, output_dir)
            run_manifest = {
                "schema_version": "aic26.keyframe_embeddings.v2",
                "model_family": args.model,
                "model_id": identity["model_id"],
                "source_drive_folder_ids": list(folder_ids),
                "source_data_root_folder_ids": list(data_root_folder_ids),
                "source_map_folder_ids": list(map_folder_ids),
                "source_mode": "shared_drive_zip_streaming",
                "video_count": video_count,
                "keyframe_count": total,
                "embedding_dimension": embedding_dim,
                "dtype": "float16",
                "l2_normalized": True,
                "preprocessing": embedder.preprocessing,
                "matrix_file": matrix_path.name,
                "metadata_file": metadata_path.name,
                "keyframe_index_file": index_path.name,
                "keyframe_maps_file": "keyframe_maps_snapshot.json",
                "mapping_source": "map-keyframes CSV (n, pts_time, fps, frame_idx)",
                "elapsed_seconds": round(time.time() - started, 3),
                "gpu": torch.cuda.get_device_name(0),
            }
            atomic_json(output_dir / "run_manifest.json", run_manifest)
            print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
            print(f"Done: {matrix_path}")
            return 0

        download_and_extract_archives(
            service,
            archives,
            data_dir,
            Path(args.zip_cache_dir),
            keep_zips=args.keep_zips,
        )
    elif data_root_folder_ids:
        raise ValueError("--skip-download cannot resolve --data-root-folder-id without Drive API")
    else:
        folder_ids = folder_ids or DEFAULT_FOLDER_IDS
        map_folder_ids = map_folder_ids or DEFAULT_MAP_FOLDER_IDS

    grouped_paths = discover_keyframes(data_dir)
    video_count = len(grouped_paths)
    image_count = sum(len(paths) for paths in grouped_paths.values())
    print(f"Discovered {image_count:,} keyframes across {video_count:,} videos")
    video_count_mismatch = video_count != args.expected_videos
    image_count_mismatch = (
        args.expected_images is not None and image_count != args.expected_images
    )
    count_mismatch = video_count_mismatch or image_count_mismatch
    if not args.allow_count_mismatch and args.max_zips is None and count_mismatch:
        expected_images = args.expected_images if args.expected_images is not None else "dynamic"
        raise RuntimeError(
            "Corpus count mismatch: "
            f"got {video_count} videos/{image_count} images; expected "
            f"{args.expected_videos}/{expected_images}. "
            "Use --allow-count-mismatch only after checking the Drive manifest."
        )
    if image_count == 0:
        raise RuntimeError(f"No keyframe images found under {data_dir}")

    identity = {
        "model_family": args.model,
        "model_id": (
            args.metaclip2_model_id if args.model == "metaclip2" else args.beit3_checkpoint_url
        ),
        "source_drive_folder_ids": list(folder_ids),
        "source_data_root_folder_ids": list(data_root_folder_ids),
        "source_map_folder_ids": list(map_folder_ids),
        "dtype": "float16",
        "l2_normalized": True,
    }
    identity_path = output_dir / "embedding_identity.json"
    ensure_embedding_identity(identity_path, identity)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; choose an A100 runtime in Colab")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)"
    )

    started = time.time()
    embedder = create_embedder(args, device)
    embedding_dim = embed_video_shards(
        grouped_paths,
        embedder,
        output_dir,
        args.batch_size,
        args.num_workers,
        device,
    )
    matrix_path, metadata_path, total = merge_outputs(
        grouped_paths,
        output_dir,
        data_dir,
        embedding_dim,
    )
    matrix = np.load(matrix_path, mmap_mode="r")
    metadata_lines = sum(1 for _ in metadata_path.open("r", encoding="utf-8"))
    if matrix.shape != (total, embedding_dim) or metadata_lines != total:
        raise RuntimeError("Final matrix/metadata verification failed")
    index_path = export_keyframe_index_csv(metadata_path, output_dir)

    run_manifest = {
        "schema_version": "aic26.keyframe_embeddings.v1",
        "model_family": args.model,
        "model_id": (
            args.metaclip2_model_id if args.model == "metaclip2" else args.beit3_checkpoint_url
        ),
        "source_drive_folder_ids": list(folder_ids),
        "source_data_root_folder_ids": list(data_root_folder_ids),
        "source_map_folder_ids": list(map_folder_ids),
        "video_count": video_count,
        "keyframe_count": total,
        "embedding_dimension": embedding_dim,
        "dtype": "float16",
        "l2_normalized": True,
        "preprocessing": embedder.preprocessing,
        "matrix_file": matrix_path.name,
        "metadata_file": metadata_path.name,
        "keyframe_index_file": index_path.name,
        "elapsed_seconds": round(time.time() - started, 3),
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    print(f"Done: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
