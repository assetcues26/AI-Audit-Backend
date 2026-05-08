"""
Integration Tests — Audit API Endpoints
Tests the full HTTP request/response cycle with mocked services.
"""
import json
from unittest.mock import patch

from app.models.schemas import ImageAuditResponse


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_name(self, client):
        response = client.get("/")
        body = response.json()
        assert "Physical Asset Audit System API" in body["message"]


class TestAuditSessionValidation:
    def test_invalid_json_metadata_returns_400(self, client, sample_image_bytes):
        files = [
            ("images", ("asset.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("barcode.jpg", sample_image_bytes, "image/jpeg")),
        ]
        data = {"session_metadata": "not-valid-json!!!"}

        response = client.post("/api/v1/audit/session", files=files, data=data)
        assert response.status_code == 400
        assert "Invalid session metadata" in response.json()["detail"]

    def test_missing_metadata_returns_422(self, client, sample_image_bytes):
        files = [
            ("images", ("asset.jpg", sample_image_bytes, "image/jpeg")),
        ]
        response = client.post("/api/v1/audit/session", files=files)
        assert response.status_code == 422

    def test_missing_images_returns_422(self, client):
        metadata = {"session_id": "s1", "pairs_metadata": [{"pair_index": 0, "barcode_skipped": False}]}
        data = {"session_metadata": json.dumps(metadata)}
        response = client.post("/api/v1/audit/session", data=data)
        assert response.status_code == 422

    def test_missing_barcode_image_returns_400(self, client, sample_image_bytes):
        """When barcode_skipped=False but no barcode image is provided."""
        files = [
            ("images", ("asset.jpg", sample_image_bytes, "image/jpeg")),
        ]
        metadata = {
            "session_id": "s1",
            "pairs_metadata": [{"pair_index": 0, "barcode_skipped": False}],
        }
        data = {"session_metadata": json.dumps(metadata)}

        with patch("app.api.v1.audit.AuditEngine.process_batch"):
            # The actual logic checks file count before calling engine
            response = client.post("/api/v1/audit/session", files=files, data=data)
            # Should get 400 because there's 1 image but 2 expected (asset + barcode)
            assert response.status_code == 400


class TestAuditSessionSuccess:
    @patch("app.api.v1.audit.AuditEngine.process_batch")
    def test_single_pair_success(self, mock_process, client, sample_image_bytes, mock_audit_result):
        mock_process.return_value = [
            ImageAuditResponse(
                pair_index=0,
                yolo_boxes=[],
                vlm_result=mock_audit_result,
            )
        ]

        files = [
            ("images", ("asset.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("barcode.jpg", sample_image_bytes, "image/jpeg")),
        ]
        metadata = {
            "session_id": "test-session-100",
            "pairs_metadata": [{"pair_index": 0, "barcode_skipped": False}],
        }

        response = client.post(
            "/api/v1/audit/session",
            files=files,
            data={"session_metadata": json.dumps(metadata)},
        )

        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["pair_index"] == 0
        assert results[0]["vlm_result"]["match_successful"] is True
        assert results[0]["vlm_result"]["asset_details"]["name"] == "Dell UltraSharp 27 Monitor"

    @patch("app.api.v1.audit.AuditEngine.process_batch")
    def test_barcode_skipped_pair(self, mock_process, client, sample_image_bytes, mock_audit_result):
        mock_process.return_value = [
            ImageAuditResponse(
                pair_index=0, yolo_boxes=[], vlm_result=mock_audit_result,
            )
        ]

        # Only 1 image since barcode is skipped
        files = [
            ("images", ("asset.jpg", sample_image_bytes, "image/jpeg")),
        ]
        metadata = {
            "session_id": "test-session-skip",
            "pairs_metadata": [{"pair_index": 0, "barcode_skipped": True}],
        }

        response = client.post(
            "/api/v1/audit/session",
            files=files,
            data={"session_metadata": json.dumps(metadata)},
        )

        assert response.status_code == 200
        assert mock_process.called

    @patch("app.api.v1.audit.AuditEngine.process_batch")
    def test_multi_pair_session(self, mock_process, client, sample_image_bytes, mock_audit_result):
        mock_process.return_value = [
            ImageAuditResponse(pair_index=i, yolo_boxes=[], vlm_result=mock_audit_result)
            for i in range(3)
        ]

        # 3 pairs: pair 0 (asset+barcode), pair 1 (asset only), pair 2 (asset+barcode)
        files = [
            ("images", ("asset0.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("barcode0.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("asset1.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("asset2.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("barcode2.jpg", sample_image_bytes, "image/jpeg")),
        ]
        metadata = {
            "session_id": "test-session-multi",
            "pairs_metadata": [
                {"pair_index": 0, "barcode_skipped": False},
                {"pair_index": 1, "barcode_skipped": True},
                {"pair_index": 2, "barcode_skipped": False},
            ],
        }

        response = client.post(
            "/api/v1/audit/session",
            files=files,
            data={"session_metadata": json.dumps(metadata)},
        )

        assert response.status_code == 200
        results = response.json()
        assert len(results) == 3


class TestAuditSessionResponseFormat:
    @patch("app.api.v1.audit.AuditEngine.process_batch")
    def test_response_contains_required_fields(self, mock_process, client, sample_image_bytes, mock_audit_result):
        mock_process.return_value = [
            ImageAuditResponse(pair_index=0, yolo_boxes=[], vlm_result=mock_audit_result)
        ]

        files = [
            ("images", ("asset.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("barcode.jpg", sample_image_bytes, "image/jpeg")),
        ]
        metadata = {
            "session_id": "format-test",
            "pairs_metadata": [{"pair_index": 0, "barcode_skipped": False}],
        }

        response = client.post(
            "/api/v1/audit/session",
            files=files,
            data={"session_metadata": json.dumps(metadata)},
        )

        result = response.json()[0]
        assert "pair_index" in result
        assert "yolo_boxes" in result
        assert "vlm_result" in result

        vlm = result["vlm_result"]
        assert "is_swapped" in vlm
        assert "match_successful" in vlm
        assert "mapping_confidence" in vlm
        assert "verification_reason" in vlm
        assert "asset_details" in vlm
        assert "barcode_details" in vlm
