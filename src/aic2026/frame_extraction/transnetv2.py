"""PyTorch TransNetV2 Shot Boundary Detection."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aic2026.contracts import ShotRecord

TRANSNET_WEIGHTS_URL = (
    "https://huggingface.co/ByteDance/shot2story/resolve/ff853c571fd92eb4e0c5713e27f2a323ac903f67/transnetv2-pytorch-weights.pth?download=true"
)
TRANSNET_WEIGHTS_FILENAME = "transnetv2-pytorch-weights.pth"


class DilatedDConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1))
        self.conv4 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(4, 0, 0), dilation=(4, 1, 1))
        self.conv5 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(8, 0, 0), dilation=(8, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x))
        x2 = F.relu(self.conv3(x1))
        x3 = F.relu(self.conv4(x2))
        x4 = F.relu(self.conv5(x3))
        return torch.cat([x, x1, x2, x3, x4], dim=1)


class TransNetV2(nn.Module):
    """Pure PyTorch TransNetV2 implementation."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.ddc1 = DilatedDConv3D(in_channels, 16)
        self.ddc2 = DilatedDConv3D(80, 16)
        self.ddc3 = DilatedDConv3D(80, 32)
        self.ddc4 = DilatedDConv3D(160, 32)

        self.spatial_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.fc1 = nn.Linear(160 * 1 * 3, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, H, W, C] -> permute to [B, C, T, H, W]
        if x.dim() == 5 and x.shape[-1] == 3:
            x = x.permute(0, 4, 1, 2, 3).float() / 255.0

        x = self.spatial_pool(self.ddc1(x))
        x = self.spatial_pool(self.ddc2(x))
        x = self.spatial_pool(self.ddc3(x))
        x = self.spatial_pool(self.ddc4(x))

        # Shape: [B, C, T, H, W] -> permute to [B, T, C, H, W] -> flatten spatial
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits, logits


@dataclass(frozen=True, slots=True)
class TransNetV2InferenceResult:
    scenes: list[tuple[int, int]]
    shots: list[ShotRecord]


def ensure_transnet_weights(cache_dir: Path | None = None) -> Path:
    target_dir = cache_dir or Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "transnetv2"
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / TRANSNET_WEIGHTS_FILENAME
    if not weights_path.exists() or weights_path.stat().st_size < 1_000_000:
        print(f"[transnetv2] Downloading PyTorch weights from {TRANSNET_WEIGHTS_URL} ...", file=sys.stderr, flush=True)
        partial = weights_path.with_suffix(".partial")
        urllib.request.urlretrieve(TRANSNET_WEIGHTS_URL, partial)
        os.replace(partial, weights_path)
        print(f"[transnetv2] Downloaded weights to {weights_path} ({weights_path.stat().st_size / 2**20:.1f} MB)", file=sys.stderr, flush=True)
    return weights_path


def load_transnetv2_model(weights_path: Path | None = None, device: str = "cuda") -> TransNetV2:
    if weights_path is None or not weights_path.exists():
        weights_path = ensure_transnet_weights()

    model = TransNetV2()
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev).eval()
    return model


def run_transnetv2_inference(
    video_path: Path,
    video_id: str,
    fps: float,
    model: TransNetV2 | None = None,
    weights_path: Path | None = None,
    batch_size: int = 16,
    threshold: float = 0.50,
    device: str = "cuda",
) -> TransNetV2InferenceResult:
    """Decode video at 48x27 thumbnails and run batched PyTorch TransNetV2 inference."""
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
    process = subprocess.Popen(
        decode_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    decoded_bytes = bytearray()
    while chunk := process.stdout.read(1024 * 1024):
        decoded_bytes.extend(chunk)
    return_code = process.wait()
    if return_code != 0:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"FFmpeg thumbnail decode failed for {video_path}: {stderr[:1000]}")

    frame_bytes = 27 * 48 * 3
    if not decoded_bytes or len(decoded_bytes) % frame_bytes:
        raise RuntimeError(f"FFmpeg emitted invalid RGB thumbnail data for {video_path}: {len(decoded_bytes)} bytes")

    frames = np.frombuffer(decoded_bytes, dtype=np.uint8).reshape((-1, 27, 48, 3)).copy()

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
    if not starts:
        starts = [0]

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    if model is None:
        model = load_transnetv2_model(weights_path=weights_path, device=str(dev))

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            inputs = np.stack([padded[start : start + 100] for start in batch_starts])
            tensor_inputs = torch.from_numpy(inputs).to(dev)
            logits, _ = model(tensor_inputs)
            predictions.append(torch.sigmoid(logits[:, 25:75, 0]).cpu().numpy())

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

    shots: list[ShotRecord] = []
    for idx, (s_start, s_end) in enumerate(scenes):
        shots.append(
            ShotRecord(
                video_id=video_id,
                shot_id=f"{video_id}:shot_{idx:04d}",
                shot_start_idx=int(s_start),
                shot_end_idx=int(s_end),
                start_time_s=float(s_start / fps),
                end_time_s=float(s_end / fps),
                fps=fps,
                source_video=str(video_path.name),
                source="transnetv2",
            )
        )

    return TransNetV2InferenceResult(scenes=scenes, shots=shots)
