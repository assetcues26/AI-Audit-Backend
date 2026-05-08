from abc import ABC, abstractmethod
from app.models.schemas import AuditResult
from typing import List

class VLMProvider(ABC):
    @abstractmethod
    async def analyze_images(self, asset_image: bytes, barcode_image: bytes, asset_boxes: List[str]) -> AuditResult:
        pass
