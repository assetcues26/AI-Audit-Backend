import pytest
import json
from unittest.mock import patch, MagicMock
from app.models.schemas import ImageAuditResponse, AuditResult

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Physical Asset Audit System API" in response.json()["message"]

def test_audit_session_endpoint_validation(client, sample_image):
    # Test with invalid metadata to ensure validation works
    files = [
        ('images', ('asset.jpg', sample_image, 'image/jpeg')),
        ('images', ('barcode.jpg', sample_image, 'image/jpeg'))
    ]
    data = {'session_metadata': 'invalid-json'}
    
    response = client.post("/api/v1/audit/session", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid session metadata" in response.json()["detail"]

@patch("app.api.v1.audit.engine.process_batch")
def test_full_audit_flow_mocked(mock_process, client, sample_image):
    # Mock the engine response to avoid API costs during testing
    mock_process.return_value = [
        ImageAuditResponse(
            pair_index=0,
            yolo_boxes=[],
            vlm_result=AuditResult(
                is_swapped=False,
                match_successful=True,
                mapping_confidence=0.95,
                verification_reason="Mocked match",
                asset_details={"name": "Test Asset", "description": "Mocked", "condition": "Good", "condition_rating": "Good"},
                barcode_details={"value": "12345", "position": "Top", "condition": "Clear", "condition_rating": "Clear"}
            )
        )
    ]

    files = [
        ('images', ('asset.jpg', sample_image, 'image/jpeg')),
        ('images', ('barcode.jpg', sample_image, 'image/jpeg'))
    ]
    
    metadata = {
        "session_id": "test-session-123",
        "pairs_metadata": [{"pair_index": 0, "barcode_skipped": False}]
    }
    
    response = client.post(
        "/api/v1/audit/session",
        files=files,
        data={"session_metadata": json.dumps(metadata)}
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["vlm_result"]["asset_details"]["name"] == "Test Asset"
    assert mock_process.called
