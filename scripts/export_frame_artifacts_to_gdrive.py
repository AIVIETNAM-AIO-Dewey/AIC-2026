#!/usr/bin/env python3
"""Mirror an organizer-compatible keyframe package to Google Drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, TypeVar

TOKEN_URI = "https://oauth2.googleapis.com/token"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
MAX_ATTEMPTS = 3
API_REQUEST_RETRIES = 3
T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--remote-root-name", default="transnetv2-only")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--preflight-only", action="store_true")
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
            "and google-auth-httplib2."
        ) from error
    return Request, Credentials, build, MediaFileUpload


def _oauth_config() -> dict[str, Any]:
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
    return config


def _folder_id() -> str:
    folder_id = os.environ.get("AIC_GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        raise ValueError("Kaggle Secret AIC_GDRIVE_FOLDER_ID is unavailable")
    return folder_id


def _credential_scopes(config: dict[str, Any]) -> list[str] | None:
    raw = config.get("scopes")
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(value, str) for value in raw):
        return raw
    raise ValueError("OAuth JSON scopes must be a string or list of strings")


def _error_label(error: Exception) -> str:
    status = getattr(getattr(error, "resp", None), "status", None)
    suffix = f" status={status}" if status is not None else ""
    return f"{type(error).__name__}{suffix}"


def _with_retry(operation: str, callback: Callable[[], T]) -> T:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(
            f"[gdrive_export] operation={operation} attempt={attempt}/{MAX_ATTEMPTS}",
            file=sys.stderr,
            flush=True,
        )
        try:
            result = callback()
        except Exception as error:
            print(
                f"[gdrive_export] operation={operation} result=failed "
                f"error={_error_label(error)}",
                file=sys.stderr,
                flush=True,
            )
            if attempt == MAX_ATTEMPTS:
                raise
            delay_s = 2 ** (attempt - 1)
            print(
                f"[gdrive_export] operation={operation} retry_in_s={delay_s}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_s)
        else:
            print(
                f"[gdrive_export] operation={operation} result=success",
                file=sys.stderr,
                flush=True,
            )
            return result
    raise RuntimeError(f"Retry loop ended unexpectedly: {operation}")


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _get_folder(service: Any, folder_id: str) -> dict[str, Any]:
    folder = _with_retry(
        "parent_folder_access",
        lambda: service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,capabilities(canAddChildren)",
            supportsAllDrives=True,
        ).execute(num_retries=0),
    )
    if folder.get("mimeType") != FOLDER_MIME_TYPE:
        raise ValueError("AIC_GDRIVE_FOLDER_ID does not point to a Google Drive folder")
    if not (folder.get("capabilities") or {}).get("canAddChildren", False):
        raise PermissionError("OAuth credential cannot add files to AIC_GDRIVE_FOLDER_ID")
    return folder


def _list_children(service: Any, folder_id: str) -> dict[str, dict[str, Any]]:
    children: dict[str, dict[str, Any]] = {}
    page_token: str | None = None
    while True:
        response = service.files().list(
            q=f"'{_escape_query(folder_id)}' in parents and trashed = false",
            fields=(
                "nextPageToken,files("
                "id,name,mimeType,size,md5Checksum,webViewLink)"
            ),
            pageSize=1000,
            pageToken=page_token,
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute(num_retries=API_REQUEST_RETRIES)
        for item in response.get("files", []):
            if item["name"] in children:
                raise ValueError(
                    f"Drive folder {folder_id} contains duplicate name: {item['name']}"
                )
            children[item["name"]] = item
        page_token = response.get("nextPageToken")
        if not page_token:
            return children


def _find_or_create_folder(
    service: Any,
    *,
    parent_id: str,
    name: str,
    children: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    existing = children.get(name)
    if existing is not None:
        if existing.get("mimeType") != FOLDER_MIME_TYPE:
            raise ValueError(f"Drive path component exists as a file: {name}")
        print(f"[gdrive_export] folder=reused name={name}", file=sys.stderr, flush=True)
        return existing
    folder = service.files().create(
        body={"name": name, "mimeType": FOLDER_MIME_TYPE, "parents": [parent_id]},
        fields="id,name,mimeType,webViewLink",
        supportsAllDrives=True,
    ).execute(num_retries=API_REQUEST_RETRIES)
    children[name] = folder
    print(f"[gdrive_export] folder=created name={name}", file=sys.stderr, flush=True)
    return folder


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_file(
    service: Any,
    media_upload_class: Any,
    *,
    path: Path,
    folder_id: str,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = media_upload_class(
        str(path),
        mimetype=mime_type,
        resumable=path.stat().st_size > 5_000_000,
    )
    fields = "id,name,mimeType,size,md5Checksum,webViewLink"
    if existing is None:
        request = service.files().create(
            body={"name": path.name, "parents": [folder_id]},
            media_body=media,
            fields=fields,
            supportsAllDrives=True,
        )
    else:
        if existing.get("mimeType") == FOLDER_MIME_TYPE:
            raise ValueError(f"Drive file path exists as a folder: {path.name}")
        request = service.files().update(
            fileId=existing["id"],
            media_body=media,
            fields=fields,
            supportsAllDrives=True,
        )
    if not media.resumable():
        return request.execute(num_retries=API_REQUEST_RETRIES)
    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=API_REQUEST_RETRIES)
        if status is not None:
            print(
                f"[gdrive_export] uploading {path.name}: {status.progress() * 100:.1f}%",
                file=sys.stderr,
                flush=True,
            )
    return response


def _build_service() -> tuple[Any, Any]:
    oauth = _oauth_config()
    print("[gdrive_export] oauth_config=loaded", file=sys.stderr, flush=True)
    Request, Credentials, build, MediaFileUpload = _drive_dependencies()
    credentials = Credentials(
        token=None,
        refresh_token=str(oauth["refresh_token"]),
        token_uri=str(oauth.get("token_uri", TOKEN_URI)),
        client_id=str(oauth["client_id"]),
        client_secret=str(oauth["client_secret"]),
        scopes=_credential_scopes(oauth),
    )
    _with_retry("oauth_refresh", lambda: credentials.refresh(Request()))
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    print("[gdrive_export] drive_client=ready", file=sys.stderr, flush=True)
    return service, MediaFileUpload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    print(
        f"[gdrive_export] api_request_retries={API_REQUEST_RETRIES}",
        file=sys.stderr,
        flush=True,
    )
    print("[gdrive_export] phase=authentication", file=sys.stderr, flush=True)
    service, MediaFileUpload = _build_service()
    parent_id = _folder_id()
    print("[gdrive_export] phase=parent_folder_preflight", file=sys.stderr, flush=True)
    parent = _get_folder(service, parent_id)
    print(
        f"[gdrive_export] parent_folder=ready name={parent['name']} writable=true",
        file=sys.stderr,
        flush=True,
    )
    if args.preflight_only:
        print(json.dumps({"status": "ready", "folder_name": parent["name"]}))
        return 0
    if args.package_root is None:
        raise ValueError("--package-root is required unless --preflight-only is used")

    package_root = args.package_root.expanduser().resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"Package root does not exist: {package_root}")
    local_files = sorted(path for path in package_root.rglob("*") if path.is_file())
    if not local_files:
        raise ValueError(f"Package root has no files: {package_root}")
    print(
        f"[gdrive_export] phase=mirror package_root={package_root} files={len(local_files)}",
        file=sys.stderr,
        flush=True,
    )

    child_cache: dict[str, dict[str, dict[str, Any]]] = {}
    parent_children = _list_children(service, parent_id)
    remote_root = _find_or_create_folder(
        service,
        parent_id=parent_id,
        name=args.remote_root_name,
        children=parent_children,
    )
    remote_folders: dict[PurePosixPath, str] = {PurePosixPath("."): remote_root["id"]}
    child_cache[remote_root["id"]] = _list_children(service, remote_root["id"])

    local_directories = sorted(
        (path for path in package_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.relative_to(package_root).parts),
    )
    for directory in local_directories:
        relative = PurePosixPath(directory.relative_to(package_root).as_posix())
        remote_parent_id = remote_folders[relative.parent]
        if remote_parent_id not in child_cache:
            child_cache[remote_parent_id] = _list_children(service, remote_parent_id)
        children = child_cache[remote_parent_id]
        remote = _find_or_create_folder(
            service,
            parent_id=remote_parent_id,
            name=relative.name,
            children=children,
        )
        remote_folders[relative] = remote["id"]
        child_cache[remote["id"]] = _list_children(service, remote["id"])

    uploaded = 0
    skipped = 0
    for index, path in enumerate(local_files, start=1):
        relative = PurePosixPath(path.relative_to(package_root).as_posix())
        remote_parent_id = remote_folders[relative.parent]
        children = child_cache[remote_parent_id]
        remote = children.get(path.name)
        if remote is not None and remote.get("md5Checksum") == _md5(path):
            skipped += 1
        else:
            remote = _upload_file(
                service,
                MediaFileUpload,
                path=path,
                folder_id=remote_parent_id,
                existing=remote,
            )
            children[path.name] = remote
            uploaded += 1
        if index == 1 or index % args.progress_every == 0 or index == len(local_files):
            print(
                f"[gdrive_export] files={index}/{len(local_files)} "
                f"uploaded={uploaded} skipped={skipped}",
                file=sys.stderr,
                flush=True,
            )

    folder_url = f"https://drive.google.com/drive/folders/{remote_root['id']}"
    print(
        json.dumps(
            {
                "status": "completed",
                "folder_id": remote_root["id"],
                "folder_url": folder_url,
                "files": len(local_files),
                "uploaded": uploaded,
                "skipped": skipped,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
