"""Adapters for external TransNetV2 scene files and inference scripts."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from collections import deque
from queue import Empty, Queue
from pathlib import Path
from threading import Thread

from aic2026.contracts import ShotRecord


def parse_scenes_txt(path: Path) -> list[tuple[int, int]]:
    scenes: list[tuple[int, int]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Invalid scene row at {path}:{line_number}: {line!r}")
            start, end = int(parts[0]), int(parts[1])
            if start < 0 or end < start:
                raise ValueError(f"Invalid scene bounds at {path}:{line_number}")
            scenes.append((start, end))
    if not scenes:
        raise ValueError(f"Scene file is empty: {path}")
    return scenes


def build_shot_records(
    *,
    video_id: str,
    scenes: list[tuple[int, int]],
    fps: float,
    source_video: Path,
) -> list[ShotRecord]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return [
        ShotRecord(
            video_id=video_id,
            shot_id=f"{video_id}:s{index:05d}",
            shot_start_idx=start,
            shot_end_idx=end,
            start_time_s=start / fps,
            end_time_s=end / fps,
            fps=fps,
            source_video=str(source_video),
        )
        for index, (start, end) in enumerate(scenes, start=1)
    ]


def run_transnetv2_inference(
    *,
    video_path: Path,
    entrypoint: Path,
    weights: Path | None,
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    linked_video = work_dir / video_path.name
    if not linked_video.exists():
        os.symlink(video_path.resolve(), linked_video)

    command = [str(entrypoint.resolve()), str(linked_video)]
    if entrypoint.suffix == ".py":
        command = [sys.executable, "-u", *command]
    if weights is not None:
        command.extend(["--weights", str(weights.resolve())])

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("$ " + " ".join(command), file=sys.stderr, flush=True)
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    lines: Queue[str | None] = Queue()
    output_tail: deque[str] = deque(maxlen=80)

    def forward_output() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    output_thread = Thread(target=forward_output, name="transnetv2-output", daemon=True)
    output_thread.start()
    started_at = time.monotonic()
    print(
        f"[transnetv2] process_started pid={process.pid}; streaming output and heartbeat every 30s",
        file=sys.stderr,
        flush=True,
    )
    while True:
        try:
            line = lines.get(timeout=30)
        except Empty:
            if process.poll() is None:
                elapsed = time.monotonic() - started_at
                print(
                    f"[transnetv2] still_running elapsed_s={elapsed:.0f}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            output_thread.join()
            continue
        if line is None:
            break
        output_tail.append(line.rstrip())
        print(line, end="", file=sys.stderr, flush=True)
    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(output_tail) or "<no child-process output captured>"
        raise RuntimeError(
            f"TransNetV2 inference failed for {video_path} with code {return_code}. "
            f"Last output:\n{tail}"
        )
    scenes_path = Path(str(linked_video) + ".scenes.txt")
    if not scenes_path.is_file():
        raise FileNotFoundError(f"TransNetV2 did not create scenes file: {scenes_path}")
    return scenes_path


def run_transnetv2_pytorch_inference(
    *,
    video_path: Path,
    model_module: Path,
    weights: Path,
    work_dir: Path,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> Path:
    """Run the official PyTorch architecture with an external `.pth` checkpoint."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if not model_module.is_file():
        raise FileNotFoundError(f"TransNetV2 PyTorch module does not exist: {model_module}")
    if not weights.is_file():
        raise FileNotFoundError(f"TransNetV2 PyTorch checkpoint does not exist: {weights}")

    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyTorch TransNetV2 requires Kaggle's torch and numpy runtime."
        ) from error

    module_spec = importlib.util.spec_from_file_location("aic2026_transnetv2_pytorch", model_module)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load TransNetV2 PyTorch module: {model_module}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model_class = getattr(module, "TransNetV2", None)
    if model_class is None:
        raise ImportError(f"TransNetV2 class is missing from PyTorch module: {model_module}")

    work_dir.mkdir(parents=True, exist_ok=True)
    decode_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path.resolve()),
        "-vf",
        "scale=48:27",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    print("$ " + " ".join(decode_command), file=sys.stderr, flush=True)
    try:
        decoded = subprocess.run(decode_command, capture_output=True, check=False)
    except FileNotFoundError as error:
        raise FileNotFoundError("ffmpeg binary is required for PyTorch TransNetV2 inference.") from error
    if decoded.returncode != 0:
        stderr = decoded.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg decode failed for {video_path}:\n{stderr}")

    frame_bytes = 27 * 48 * 3
    if not decoded.stdout or len(decoded.stdout) % frame_bytes:
        raise RuntimeError(
            f"ffmpeg emitted invalid RGB thumbnail data for {video_path}: {len(decoded.stdout)} bytes"
        )
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape((-1, 27, 48, 3)).copy()
    print(f"[transnetv2:pytorch] decoded_frames={len(frames)}", file=sys.stderr, flush=True)

    remainder = len(frames) % 50
    end_padding = 25 + 50 - (remainder if remainder else 50)
    padded = np.concatenate(
        [
            np.repeat(frames[:1], 25, axis=0),
            frames,
            np.repeat(frames[-1:], end_padding, axis=0),
        ],
        axis=0,
    )
    starts = list(range(0, len(padded) - 99, 50))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transnetv2:pytorch] device={device} windows={len(starts)}", file=sys.stderr, flush=True)
    model = model_class()
    try:
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()

    predictions: list[object] = []
    with torch.no_grad():
        for batch_number, offset in enumerate(range(0, len(starts), batch_size), start=1):
            batch_starts = starts[offset : offset + batch_size]
            inputs = np.stack([padded[start : start + 100] for start in batch_starts])
            logits, _ = model(torch.from_numpy(inputs).to(device))
            predictions.append(torch.sigmoid(logits[:, 25:75, 0]).cpu().numpy())
            if batch_number == 1 or batch_number % 16 == 0 or offset + batch_size >= len(starts):
                completed = min(offset + batch_size, len(starts))
                print(
                    f"[transnetv2:pytorch] inferred_windows={completed}/{len(starts)}",
                    file=sys.stderr,
                    flush=True,
                )
    scores = np.concatenate(predictions, axis=0).reshape(-1)[: len(frames)]
    boundaries = (scores > threshold).astype(np.uint8)
    scenes: list[tuple[int, int]] = []
    previous, start = 0, 0
    for index, boundary in enumerate(boundaries):
        if previous == 1 and boundary == 0:
            start = index
        if previous == 0 and boundary == 1 and index != 0:
            scenes.append((start, index))
        previous = int(boundary)
    if previous == 0:
        scenes.append((start, len(boundaries) - 1))
    if not scenes:
        scenes = [(0, len(frames) - 1)]

    scenes_path = work_dir / f"{video_path.name}.scenes.txt"
    np.savetxt(scenes_path, np.asarray(scenes, dtype=np.int32), fmt="%d")
    print(f"[transnetv2:pytorch] scenes={len(scenes)} output={scenes_path}", file=sys.stderr, flush=True)
    return scenes_path
