from typing import List
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
import ultralytics.nn.tasks
from ultralytics import YOLO

from app.core.config import settings
from app.models.schemas import BoundingBox

# Fix for PyTorch 2.6+ unpickling issue with Ultralytics models
try:
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        torch.nn.modules.container.Sequential,
    ])
except AttributeError:
    # Older torch versions don't have this method
    pass

class YOLOService:
    def __init__(self):
        self.model = None

    def _load_model(self) -> None:
        if self.model is None:
            self.model = YOLO(settings.YOLO_MODEL)

    def detect_objects(self, image_bytes: bytes) -> List[BoundingBox]:
        self._load_model()

        import io

        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        results = self.model(img)

        boxes = []
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].tolist()  # x1, y1, x2, y2
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names[cls]
                boxes.append(BoundingBox(
                    x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                    confidence=conf, class_name=name
                ))
        return boxes
