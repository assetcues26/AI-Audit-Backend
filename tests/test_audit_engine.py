"""
Integration Tests — Audit Engine
Tests the AuditEngine orchestration with mocked YOLO + VLM services.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.audit_engine import AuditEngine
from app.models.schemas import AuditResult, BoundingBox, ImageAuditResponse


class TestAuditEngineProcessBatch:
    @pytest.mark.asyncio
    @patch("app.services.audit_engine.LocalStorageAdapter")
    @patch("app.services.audit_engine.VLMService")
    @patch("app.services.audit_engine.YOLOService")
    async def test_single_pair_processing(
        self, MockYOLO, MockVLM, MockStorage, mock_audit_result, sample_image_bytes
    ):
        # Setup mocks
        mock_yolo_instance = MockYOLO.return_value
        mock_yolo_instance.detect_objects.return_value = [
            BoundingBox(x1=10, y1=20, x2=200, y2=300, confidence=0.9, class_name="laptop")
        ]

        mock_vlm_instance = MockVLM.return_value
        mock_vlm_instance.analyze_images = AsyncMock(return_value=mock_audit_result)

        mock_storage_instance = MockStorage.return_value

        engine = AuditEngine()
        pairs = [{"asset": sample_image_bytes, "barcode": sample_image_bytes}]
        results = await engine.process_batch("test-session", pairs)

        assert len(results) == 1
        assert isinstance(results[0], ImageAuditResponse)
        assert results[0].vlm_result.match_successful is True
        assert len(results[0].yolo_boxes) == 1
        assert results[0].yolo_boxes[0].class_name == "laptop"
        mock_storage_instance.save_session.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.audit_engine.LocalStorageAdapter")
    @patch("app.services.audit_engine.VLMService")
    @patch("app.services.audit_engine.YOLOService")
    async def test_multiple_pairs(
        self, MockYOLO, MockVLM, MockStorage, mock_audit_result, sample_image_bytes
    ):
        mock_yolo_instance = MockYOLO.return_value
        mock_yolo_instance.detect_objects.return_value = []

        mock_vlm_instance = MockVLM.return_value
        mock_vlm_instance.analyze_images = AsyncMock(return_value=mock_audit_result)

        engine = AuditEngine()
        pairs = [
            {"asset": sample_image_bytes, "barcode": sample_image_bytes},
            {"asset": sample_image_bytes, "barcode": b""},
            {"asset": sample_image_bytes, "barcode": sample_image_bytes},
        ]

        results = await engine.process_batch("multi-session", pairs)
        assert len(results) == 3
        assert all(isinstance(r, ImageAuditResponse) for r in results)

    @pytest.mark.asyncio
    @patch("app.services.audit_engine.LocalStorageAdapter")
    @patch("app.services.audit_engine.VLMService")
    @patch("app.services.audit_engine.YOLOService")
    async def test_yolo_error_handled_gracefully(
        self, MockYOLO, MockVLM, MockStorage, sample_image_bytes
    ):
        """If YOLO crashes on a pair, the engine should catch the error
        and return an error AuditResult instead of crashing entirely."""
        mock_yolo_instance = MockYOLO.return_value
        mock_yolo_instance.detect_objects.side_effect = Exception("YOLO model not loaded")

        engine = AuditEngine()
        pairs = [{"asset": sample_image_bytes, "barcode": sample_image_bytes}]

        results = await engine.process_batch("error-session", pairs)
        assert len(results) == 1
        assert results[0].vlm_result.match_successful is False
        assert "Processing error" in results[0].vlm_result.verification_reason

    @pytest.mark.asyncio
    @patch("app.services.audit_engine.LocalStorageAdapter")
    @patch("app.services.audit_engine.VLMService")
    @patch("app.services.audit_engine.YOLOService")
    async def test_vlm_error_handled_gracefully(
        self, MockYOLO, MockVLM, MockStorage, sample_image_bytes
    ):
        """If the VLM service crashes, the engine should catch it gracefully."""
        mock_yolo_instance = MockYOLO.return_value
        mock_yolo_instance.detect_objects.return_value = []

        mock_vlm_instance = MockVLM.return_value
        mock_vlm_instance.analyze_images = AsyncMock(
            side_effect=Exception("Gemini API quota exceeded")
        )

        engine = AuditEngine()
        pairs = [{"asset": sample_image_bytes, "barcode": sample_image_bytes}]

        results = await engine.process_batch("vlm-error-session", pairs)
        assert len(results) == 1
        assert results[0].vlm_result.match_successful is False
        assert "Gemini API quota exceeded" in results[0].vlm_result.verification_reason

    @pytest.mark.asyncio
    @patch("app.services.audit_engine.LocalStorageAdapter")
    @patch("app.services.audit_engine.VLMService")
    @patch("app.services.audit_engine.YOLOService")
    async def test_session_saved_to_storage(
        self, MockYOLO, MockVLM, MockStorage, mock_audit_result, sample_image_bytes
    ):
        """Verify that the engine persists results via the storage adapter."""
        mock_yolo_instance = MockYOLO.return_value
        mock_yolo_instance.detect_objects.return_value = []

        mock_vlm_instance = MockVLM.return_value
        mock_vlm_instance.analyze_images = AsyncMock(return_value=mock_audit_result)

        mock_storage_instance = MockStorage.return_value

        engine = AuditEngine()
        pairs = [{"asset": sample_image_bytes, "barcode": sample_image_bytes}]

        await engine.process_batch("save-test-session", pairs)

        mock_storage_instance.save_session.assert_called_once()
        call_args = mock_storage_instance.save_session.call_args
        assert call_args[0][0] == "save-test-session"
        assert isinstance(call_args[0][1], list)

    @pytest.mark.asyncio
    @patch("app.services.audit_engine.LocalStorageAdapter")
    @patch("app.services.audit_engine.VLMService")
    @patch("app.services.audit_engine.YOLOService")
    async def test_empty_batch(self, MockYOLO, MockVLM, MockStorage):
        """An empty batch should return empty results without crashing."""
        engine = AuditEngine()
        results = await engine.process_batch("empty-session", [])
        assert results == []
