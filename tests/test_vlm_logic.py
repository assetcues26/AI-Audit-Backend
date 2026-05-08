"""
Unit Tests — VLM Logic (Gemini Adapter)
Tests prompt generation, JSON extraction, and error fallback behavior.
All external API calls are mocked.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.vlm_providers.gemini_adapter import GeminiAdapter
from app.models.schemas import AuditResult


class TestPromptGeneration:
    @patch("app.services.vlm_providers.gemini_adapter.genai")
    @patch("app.services.vlm_providers.gemini_adapter.settings")
    def test_prompt_includes_yolo_boxes(self, mock_settings, mock_genai):
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.VLM_MODEL = "test-model"
        adapter = GeminiAdapter()
        boxes = ["chair", "laptop", "monitor"]
        prompt = adapter._get_prompt(boxes)

        assert "chair, laptop, monitor" in prompt
        assert "STEP 1" in prompt
        assert "SWAP DETECTION" in prompt

    @patch("app.services.vlm_providers.gemini_adapter.genai")
    @patch("app.services.vlm_providers.gemini_adapter.settings")
    def test_prompt_empty_boxes(self, mock_settings, mock_genai):
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.VLM_MODEL = "test-model"
        adapter = GeminiAdapter()
        prompt = adapter._get_prompt([])

        assert "YOLO detected no specific regions" in prompt

    @patch("app.services.vlm_providers.gemini_adapter.genai")
    @patch("app.services.vlm_providers.gemini_adapter.settings")
    def test_prompt_contains_output_format(self, mock_settings, mock_genai):
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.VLM_MODEL = "test-model"
        adapter = GeminiAdapter()
        prompt = adapter._get_prompt(["desk"])

        assert "OUTPUT FORMAT" in prompt
        assert "is_swapped" in prompt
        assert "match_successful" in prompt


class TestJSONExtraction:
    def test_extract_json_from_wrapped_text(self):
        """Simulate Gemini returning JSON wrapped in conversational text."""
        raw_text = """
        Here is your audit result:
        {
          "is_swapped": false,
          "match_successful": true,
          "mapping_confidence": 0.9,
          "verification_reason": "Everything matches.",
          "asset_details": {
            "name": "Office Chair",
            "description": "Ergonomic mesh chair",
            "condition": "Good — no damage",
            "condition_rating": "Good"
          },
          "barcode_details": {
            "value": "BAR-001",
            "position": "Base of chair",
            "condition": "Clear — intact label",
            "condition_rating": "Clear"
          }
        }
        Hope this helps!
        """
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        json_str = raw_text[start:end]
        data = json.loads(json_str)

        assert data["asset_details"]["name"] == "Office Chair"
        assert data["match_successful"] is True
        assert data["mapping_confidence"] == 0.9

    def test_extract_json_clean_response(self):
        """Simulate Gemini returning clean JSON only."""
        raw_text = '{"is_swapped": false, "match_successful": true, "mapping_confidence": 1.0, "verification_reason": "Match", "asset_details": {"name": "Laptop"}, "barcode_details": {"value": "X"}}'
        data = json.loads(raw_text)
        assert data["asset_details"]["name"] == "Laptop"

    def test_no_json_found(self):
        """Simulate a response with no JSON at all."""
        raw_text = "I cannot identify the asset in this image."
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        assert start == -1 or end == 0


class TestAnalyzeImagesFallback:
    @pytest.mark.asyncio
    @patch("app.services.vlm_providers.gemini_adapter.genai")
    @patch("app.services.vlm_providers.gemini_adapter.settings")
    async def test_parsing_error_returns_fallback(self, mock_settings, mock_genai):
        """When Gemini returns un-parseable text, the adapter should return a
        graceful fallback AuditResult instead of crashing."""
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.VLM_MODEL = "test-model"
        adapter = GeminiAdapter()

        mock_response = MagicMock()
        mock_response.text = "Sorry, I could not process this request."

        with patch.object(adapter.model, "generate_content", return_value=mock_response):
            with patch(
                "app.services.vlm_providers.gemini_adapter.process_image_for_vlm",
                return_value=MagicMock(),
            ):
                result = await adapter.analyze_images(b"fake-asset", b"fake-barcode", [])

        assert isinstance(result, AuditResult)
        assert result.match_successful is False
        assert result.asset_details.name == "Unknown"
        assert "AI Parsing Error" in result.verification_reason

    @pytest.mark.asyncio
    @patch("app.services.vlm_providers.gemini_adapter.genai")
    @patch("app.services.vlm_providers.gemini_adapter.settings")
    async def test_successful_parse(self, mock_settings, mock_genai):
        """When Gemini returns valid JSON, the adapter should parse it correctly."""
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.VLM_MODEL = "test-model"
        adapter = GeminiAdapter()

        valid_json = json.dumps({
            "is_swapped": False,
            "match_successful": True,
            "mapping_confidence": 0.85,
            "verification_reason": "All checks passed.",
            "asset_details": {
                "name": "HP LaserJet Printer",
                "description": "Compact office printer",
                "condition": "Good — clean exterior",
                "condition_rating": "Good",
            },
            "barcode_details": {
                "value": "HP-LJ-2024",
                "position": "Right side panel",
                "condition": "Clear — fully legible",
                "condition_rating": "Clear",
            },
        })

        mock_response = MagicMock()
        mock_response.text = valid_json

        with patch.object(adapter.model, "generate_content", return_value=mock_response):
            with patch(
                "app.services.vlm_providers.gemini_adapter.process_image_for_vlm",
                return_value=MagicMock(),
            ):
                result = await adapter.analyze_images(b"fake-asset", b"fake-barcode", ["printer"])

        assert result.match_successful is True
        assert result.asset_details.name == "HP LaserJet Printer"
        assert result.mapping_confidence == 0.85

    @pytest.mark.asyncio
    @patch("app.services.vlm_providers.gemini_adapter.genai")
    @patch("app.services.vlm_providers.gemini_adapter.settings")
    async def test_empty_barcode_image(self, mock_settings, mock_genai):
        """When barcode image is empty (skipped), the adapter should still work."""
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.VLM_MODEL = "test-model"
        adapter = GeminiAdapter()

        valid_json = json.dumps({
            "is_swapped": False,
            "match_successful": False,
            "mapping_confidence": 0.5,
            "verification_reason": "No barcode provided.",
            "asset_details": {"name": "Desk Lamp"},
            "barcode_details": {"value": None},
        })

        mock_response = MagicMock()
        mock_response.text = valid_json

        with patch.object(adapter.model, "generate_content", return_value=mock_response):
            with patch(
                "app.services.vlm_providers.gemini_adapter.process_image_for_vlm",
                return_value=MagicMock(),
            ):
                result = await adapter.analyze_images(b"fake-asset", b"", [])

        assert result.asset_details.name == "Desk Lamp"
        assert result.barcode_details.value is None
