from typing import List, Optional

from pydantic import BaseModel


class AssetDetails(BaseModel):
    name: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[str] = None
    condition_rating: Optional[str] = None

class BarcodeDetails(BaseModel):
    value: Optional[str] = None
    position: Optional[str] = None
    condition: Optional[str] = None
    condition_rating: Optional[str] = None

class AuditResult(BaseModel):
    is_swapped: bool = False
    match_successful: bool = False
    asset_details: AssetDetails
    barcode_details: BarcodeDetails
    mapping_confidence: float
    verification_reason: str

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

class ImageAuditResponse(BaseModel):
    pair_index: int
    yolo_boxes: List[BoundingBox] = []
    vlm_result: AuditResult
