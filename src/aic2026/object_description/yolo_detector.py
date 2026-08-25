"""Open-vocabulary object detector adapter using Ultralytics YOLO-World."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from .geometry import Detection


DEFAULT_OPEN_VOCABULARY = [
    # People & Attire
    "person", "man", "woman", "child", "crowd",
    # Vehicles & Transportation
    "car", "vehicle", "automobile", "motorcycle", "motorbike", "bicycle", "bike",
    "bus", "truck", "van", "boat", "ship", "airplane", "train",
    # Infrastructure, Buildings & Architecture
    "building", "house", "roof", "bridge", "road", "street", "highway", "sidewalk",
    "traffic light", "street sign", "billboard", "banner", "fence", "wall", "gate",
    "door", "window", "pole", "power line",
    # Nature, Landforms & Hazards
    "tree", "plant", "flower", "grass", "water", "river", "canal", "lake", "sea",
    "mountain", "hill", "landslide", "road collapse", "flood", "mud", "sky",
    "fire", "smoke",
    # Indoor & Everyday Objects
    "chair", "table", "desk", "sofa", "bed", "television", "tv screen", "monitor",
    "laptop", "computer", "phone", "cell phone", "camera", "clock",
    "bottle", "cup", "glass", "plate", "bowl", "food", "fruit",
    "backpack", "bag", "handbag", "suitcase", "umbrella", "hat", "helmet", "glasses", "sunglasses",
    # Salient / Catch-all
    "sign", "logo", "poster", "document", "text", "animal", "dog", "cat", "bird", "object",
]


class YoloWorldDetector:
    """YOLO-World open-vocabulary detector generating normalized Detections."""

    def __init__(
        self,
        model_id_or_path: str | Path = "yolov8x-worldv2.pt",
        *,
        device: str = "auto",
        conf: float = 0.15,
        custom_classes: list[str] | str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO-World detection. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.model_path = str(model_id_or_path)
        self.conf = float(conf)
        self.device = device
        
        # Initialize YOLO-World model
        self.model = YOLOWorld(self.model_path)
        
        # Set open-vocabulary classes
        if custom_classes is None:
            self.model.set_classes(DEFAULT_OPEN_VOCABULARY)
        elif isinstance(custom_classes, list) and custom_classes:
            self.model.set_classes(custom_classes)

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect objects in a PIL Image and return normalized Detection records."""
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size
        if w <= 0 or h <= 0:
            return []

        # Predict with YOLO-World
        results = self.model.predict(
            source=image_rgb,
            conf=self.conf,
            device=self.device if self.device != "auto" else None,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names_dict = result.names or {}

        for idx in range(len(boxes)):
            try:
                # xyxy coordinates in pixels
                xyxy = boxes.xyxy[idx].tolist()
                score = float(boxes.conf[idx].item())
                cls_id = int(boxes.cls[idx].item())
                class_name = str(names_dict.get(cls_id, f"object_{cls_id}")).strip() or "object"

                x1, y1, x2, y2 = xyxy

                # Convert to normalized ymin, xmin, ymax, xmax
                ymin_norm = max(0.0, min(1.0, float(y1) / h))
                xmin_norm = max(0.0, min(1.0, float(x1) / w))
                ymax_norm = max(0.0, min(1.0, float(y2) / h))
                xmax_norm = max(0.0, min(1.0, float(x2) / w))

                if ymax_norm <= ymin_norm or xmax_norm <= xmin_norm:
                    continue

                detections.append(
                    Detection(
                        source_index=idx,
                        score=score,
                        class_name=class_name,
                        class_entity=class_name,
                        class_label=cls_id,
                        bbox_yxyx_norm=(ymin_norm, xmin_norm, ymax_norm, xmax_norm),
                    )
                )
            except Exception:
                continue

        return detections
