"""
Unit Tests — Pydantic Schemas
Tests that all data models validate correctly and reject bad data.
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    AssetDetails, BarcodeDetails, AuditResult,
    BoundingBox, ImageAuditResponse,
)


# ──────────────────────────────────────────────
# AssetDetails
# ──────────────────────────────────────────────
class TestAssetDetails:
    def test_full_asset_details(self):
        detail = AssetDetails(
            name="Herman Miller Aeron Chair",
            id="ASSET-042",
            description="Ergonomic mesh office chair with lumbar support.",
            condition="Good — no visible damage.",
            condition_rating="Good",
        )
        assert detail.name == "Herman Miller Aeron Chair"
        assert detail.condition_rating == "Good"

    def test_all_fields_optional(self):
        detail = AssetDetails()
        assert detail.name is None
        assert detail.id is None
        assert detail.description is None
        assert detail.condition is None
        assert detail.condition_rating is None

    def test_partial_fields(self):
        detail = AssetDetails(name="Desk Lamp")
        assert detail.name == "Desk Lamp"
        assert detail.id is None


# ──────────────────────────────────────────────
# BarcodeDetails
# ──────────────────────────────────────────────
class TestBarcodeDetails:
    def test_full_barcode_details(self):
        detail = BarcodeDetails(
            value="QR-ABC-123",
            position="Top-right corner of the chassis.",
            condition="Clear — no peeling or damage.",
            condition_rating="Clear",
        )
        assert detail.value == "QR-ABC-123"
        assert detail.condition_rating == "Clear"

    def test_all_fields_optional(self):
        detail = BarcodeDetails()
        assert detail.value is None
        assert detail.position is None


# ──────────────────────────────────────────────
# BoundingBox
# ──────────────────────────────────────────────
class TestBoundingBox:
    def test_valid_bounding_box(self):
        box = BoundingBox(x1=10, y1=20, x2=200, y2=300, confidence=0.92, class_name="laptop")
        assert box.class_name == "laptop"
        assert box.confidence == 0.92

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=20)  # Missing x2, y2, confidence, class_name

    def test_bounding_box_float_precision(self):
        box = BoundingBox(
            x1=10.123456, y1=20.654321,
            x2=200.111, y2=300.999,
            confidence=0.876543, class_name="mouse",
        )
        assert isinstance(box.confidence, float)


# ──────────────────────────────────────────────
# AuditResult
# ──────────────────────────────────────────────
class TestAuditResult:
    def test_valid_audit_result(self, mock_audit_result):
        assert mock_audit_result.match_successful is True
        assert mock_audit_result.mapping_confidence == 0.95
        assert mock_audit_result.asset_details.name == "Dell UltraSharp 27 Monitor"

    def test_audit_result_with_dict_details(self):
        """Ensure dict-based details are auto-parsed into Pydantic models."""
        result = AuditResult(
            is_swapped=False,
            match_successful=False,
            mapping_confidence=0.0,
            verification_reason="Test reason",
            asset_details={"name": "Chair", "id": None, "description": "A chair", "condition": None},
            barcode_details={"value": None, "position": None, "condition": None},
        )
        assert isinstance(result.asset_details, AssetDetails)
        assert result.asset_details.name == "Chair"

    def test_audit_result_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AuditResult()  # Missing all required fields


# ──────────────────────────────────────────────
# ImageAuditResponse
# ──────────────────────────────────────────────
class TestImageAuditResponse:
    def test_valid_response(self, mock_audit_result, mock_bounding_boxes):
        response = ImageAuditResponse(
            pair_index=0,
            yolo_boxes=mock_bounding_boxes,
            vlm_result=mock_audit_result,
        )
        assert response.pair_index == 0
        assert len(response.yolo_boxes) == 2
        assert response.vlm_result.match_successful is True

    def test_empty_yolo_boxes(self, mock_audit_result):
        response = ImageAuditResponse(
            pair_index=1,
            yolo_boxes=[],
            vlm_result=mock_audit_result,
        )
        assert response.yolo_boxes == []

    def test_model_dump_serialization(self, mock_audit_result):
        response = ImageAuditResponse(
            pair_index=0, yolo_boxes=[], vlm_result=mock_audit_result,
        )
        data = response.model_dump()
        assert isinstance(data, dict)
        assert "vlm_result" in data
        assert data["vlm_result"]["asset_details"]["name"] == "Dell UltraSharp 27 Monitor"
