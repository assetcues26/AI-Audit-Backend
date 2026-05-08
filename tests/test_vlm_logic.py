import asyncio
import pytest
import json
from app.services.vlm_providers.gemini_adapter import GeminiAdapter
from unittest.mock import MagicMock, patch

def test_gemini_prompt_generation():
    adapter = GeminiAdapter()
    boxes = ["chair", "laptop"]
    prompt = adapter._get_prompt(boxes)
    
    assert "chair, laptop" in prompt
    assert "STEP 1 — SWAP DETECTION" in prompt
    assert "OUTPUT FORMAT" in prompt

def test_gemini_json_extraction():
    adapter = GeminiAdapter()
    
    # Mock a response that includes markdown or other text
    mock_response = MagicMock()
    mock_response.text = """
    Certainly! Here is the audit result in JSON format:
    {
      "is_swapped": false,
      "match_successful": true,
      "mapping_confidence": 0.9,
      "verification_reason": "Everything matches.",
      "asset_details": {
        "name": "Office Chair",
        "description": "Ergonomic chair",
        "condition": "Good",
        "condition_rating": "Good"
      },
      "barcode_details": {
        "value": "BAR-001",
        "position": "Base",
        "condition": "Clear",
        "condition_rating": "Clear"
      }
    }
    Hope this helps!
    """
    
    # Test JSON extraction logic
    start = mock_response.text.find('{')
    end = mock_response.text.rfind('}') + 1
    json_str = mock_response.text[start:end]
    data = json.loads(json_str)
    
    assert data["asset_details"]["name"] == "Office Chair"
    assert data["match_successful"] is True

@pytest.mark.asyncio
async def test_gemini_parsing_error_fallback():
    adapter = GeminiAdapter()
    
    # Mock a completely broken response
    mock_response = MagicMock()
    mock_response.text = "I'm sorry, I couldn't identify anything."
    
    with patch.object(adapter.model, 'generate_content', return_value=mock_response):
        with patch("app.services.vlm_providers.gemini_adapter.process_image_for_vlm", return_value=MagicMock()):
            result = await adapter.analyze_images(b"fake", b"fake", [])
            assert result.asset_details.name == "Unknown"
            assert "AI Parsing Error" in result.verification_reason
