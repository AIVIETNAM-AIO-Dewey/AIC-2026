#!/usr/bin/env python3
"""Upload adaptive frame artifacts to a browsable Google Drive folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
TOKEN_URI = "https://oauth2.googleapis.com/token"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folder-name")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _drive_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Google Drive export requires google-api-python-client, google-auth, "
            "and google-auth-httplib2. Install them in the Kaggle notebook export cell."
        ) from error
    return Request, Credentials, build, MediaFileUpload


def _oauth_config() -> dict[str, str]:
    raw = os.environ.get("AIC_GDRIVE_OAUTH_JSON", "").strip()
    if not raw:
        raise ValueError("Kaggle Secret AIC_GDRIVE_OAUTH_JSON is unavailable")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("AIC_GDRIVE_OAUTH_JSON must contain valid JSON") from error
    required = ("client_id", "client_secret", "refresh_token")
    missing = [key for key in required if not str(config.get(key, "")).strip()]
    if missing:
        raise ValueError(f"AIC_GDRIVE_OAUTH_JSON is missing keys: {missing}")
    return {key: str(value) for key, value in config.items() if value is not None}


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _find_or_create_folder(service: Any, *, name: str, parent_id: str | None) -> dict[str, Any]:
    parent_query = f"'{_escape_query(parent_id)}' in parents" if parent_id else "'root' in parents"
    query = (
        f"name = '{_escape_query(name)}' and mimeType = '{FOLDER_MIME_TYPE}' "
        f"and {parent_query} and trashed = false"
    )
    response = service.files().list(
        q=query,
        fields="files(id,name,webViewLink)",
        spaces="drive",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute(num_retries=3)
    matches = response.get("files", [])
    if matches:
        return matches[0]
    body: dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME_TYPE}
    if parent_id:
        body["parents"] = [parent_id]
    return service.files().create(
        body=body,
        fields="id,name,webViewLink",
        supportsAllDrives=True,
    ).execute(num_retries=3)


def _list_children(service: Any, folder_id: str) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    page_token: str | None = None
    while True:
        response = service.files().list(
            q=f"'{_escape_query(folder_id)}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,size,md5Checksum,webViewLink)",
            pageSize=1000,
            pageToken=page_token,
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute(num_retries=3)
        for item in response.get("files", []):
            files[item["name"]] = item
        page_token = response.get("nextPageToken")
        if not page_token:
            return files


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _upload(
    service: Any,
    media_upload_class: Any,
    *,
    path: Path,
    folder_id: str,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = media_upload_class(str(path), mimetype=mime_type, resumable=path.stat().st_size > 5_000_000)
    fields = "id,name,size,md5Checksum,webViewLink"
    if existing is None:
        request = service.files().create(
            body={"name": path.name, "parents": [folder_id]},
            media_body=media,
            fields=fields,
            supportsAllDrives=True,
        )
    else:
        request = service.files().update(
            fileId=existing["id"],
            media_body=media,
            fields=fields,
            supportsAllDrives=True,
        )
    if not media.resumable():
        return request.execute(num_retries=3)
    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=3)
        if status is not None:
            print(
                f"[gdrive_export] uploading {path.name}: {status.progress() * 100:.1f}%",
                file=sys.stderr,
                flush=True,
            )
    return response


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    oauth = _oauth_config()
    Request, Credentials, build, MediaFileUpload = _drive_dependencies()
    credentials = Credentials(
        token=None,
        refresh_token=oauth["refresh_token"],
        token_uri=oauth.get("token_uri", TOKEN_URI),
        client_id=oauth["client_id"],
        client_secret=oauth["client_secret"],
        scopes=[DRIVE_SCOPE],
    )
    credentials.refresh(Request())
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    output_root = args.output_root.expanduser().resolve()
    frames_dir = output_root / "adaptive_keyframes" / args.video_id
    adaptive_manifest = (
        output_root / "frame_extraction" / "adaptive_manifests" / f"{args.video_id}.jsonl"
    )
    comparison_dir = output_root / "frame_extraction" / "comparison" / args.video_id
    if not frames_dir.is_dir() or not adaptive_manifest.is_file():
        raise FileNotFoundError("Adaptive frames and manifest must be extracted before Drive export")
    upload_paths = sorted(frames_dir.glob("*.jpg"))
    upload_paths.append(adaptive_manifest)
    for name in (
        "summary.json",
        "organizer_samples.jsonl",
        "transnetv2_adaptive_samples.jsonl",
        "transnetv2_additions.jsonl",
        "merged_samples.jsonl",
    ):
        path = comparison_dir / name
        if path.is_file():
            upload_paths.append(path)

    folder_name = args.folder_name or f"AIC2026-{args.video_id}-adaptive-frames"
    parent_id = oauth.get("folder_id", "").strip() or None
    folder = _find_or_create_folder(service, name=folder_name, parent_id=parent_id)
    existing = _list_children(service, folder["id"])
    uploaded = 0
    skipped = 0
    for index, path in enumerate(upload_paths, start=1):
        remote = existing.get(path.name)
        if remote is not None and remote.get("md5Checksum") == _md5(path):
            skipped += 1
        else:
            remote = _upload(
                service,
                MediaFileUpload,
                path=path,
                folder_id=folder["id"],
                existing=remote,
            )
            existing[path.name] = remote
            uploaded += 1
        if index == 1 or index % args.progress_every == 0 or index == len(upload_paths):
            print(
                f"[gdrive_export] files={index}/{len(upload_paths)} "
                f"uploaded={uploaded} skipped={skipped}",
                file=sys.stderr,
                flush=True,
            )

    folder_url = f"https://drive.google.com/drive/folders/{folder['id']}"
    print(
        json.dumps(
            {
                "status": "completed",
                "folder_id": folder["id"],
                "folder_url": folder_url,
                "uploaded": uploaded,
                "skipped": skipped,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
