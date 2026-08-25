"""Open-vocabulary object detector adapter using Ultralytics YOLO-World."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from .geometry import Detection


DEFAULT_OPEN_VOCABULARY = [
    # 1. People, Roles & Body Parts
    "person", "man", "woman", "child", "girl", "boy", "crowd",
    "chef", "news anchor", "reporter", "host", "police officer", "doctor", "athlete",
    "human face", "human hand",

    # 2. Clothing, Attire & Accessories
    "clothing", "shirt", "jacket", "chef jacket", "suit", "dress", "t-shirt", "coat",
    "apron", "tie", "pants", "shorts", "shoes",
    "glasses", "sunglasses", "hat", "cap", "helmet", "gloves", "mask",
    "backpack", "bag", "handbag", "suitcase", "watch",

    # 3. Vehicles & Transportation
    "car", "automobile", "vehicle", "motorcycle", "motorbike", "scooter", "bicycle", "bike",
    "bus", "truck", "van", "ambulance", "fire truck",
    "boat", "ship", "canoe", "airplane", "helicopter", "train",
    "license plate", "traffic light", "traffic sign",

    # 4. Urban, Infrastructure, Architecture & Hazards
    "building", "house", "roof", "window", "door", "gate", "wall", "fence", "balcony",
    "bridge", "road", "street", "highway", "sidewalk", "intersection", "pavement",
    "electric pole", "billboard", "banner", "poster", "flag", "signboard",
    "landslide", "road collapse", "flood", "mud", "fire", "smoke",

    # 5. Nature, Environment & Water
    "tree", "plant", "flower", "grass", "leaves", "forest",
    "water", "river", "canal", "lake", "pond", "sea", "beach",
    "mountain", "hill", "sky", "cloud",

    # 6. Food, Cooking & Kitchenware
    "food", "dish", "meal", "vegetable", "tomato", "salad", "fruit", "meat", "fish", "soup", "snack",
    "bowl", "plate", "dish", "cup", "glass", "bottle",
    "frying pan", "pot", "pan", "stove", "cutting board", "tableware", "spoon", "fork", "knife", "chopsticks",

    # 7. Studio, Media, Electronics & Office
    "desk", "table", "chair", "sofa", "podium",
    "microphone", "camera", "tv screen", "television", "monitor", "laptop", "computer", "phone", "smartphone",
    "logo", "text", "document", "paper", "book", "clock",

    # 8. Animals
    "dog", "cat", "bird", "fish", "horse", "cow", "chicken", "animal",
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
