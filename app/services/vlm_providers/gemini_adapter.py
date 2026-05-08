import logging
import json
import google.generativeai as genai
from typing import List
from app.core.config import settings
from app.models.schemas import AuditResult
from app.services.vlm_providers.base import VLMProvider
from app.utils.image_utils import process_image_for_vlm

logger = logging.getLogger(__name__)

class GeminiAdapter(VLMProvider):
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.VLM_MODEL)
        logger.info(f"GeminiAdapter initialized with model: {settings.VLM_MODEL}")

    async def analyze_images(self, asset_image: bytes, barcode_image: bytes, asset_boxes: List[str]) -> AuditResult:
        prompt = self._get_prompt(asset_boxes)
        img1 = process_image_for_vlm(asset_image)
        
        content_parts = [prompt, img1]
        
        # Only add the second image if it's not empty
        if barcode_image and len(barcode_image) > 0:
            logger.info("Including barcode image in VLM request")
            img2 = process_image_for_vlm(barcode_image)
            content_parts.append(img2)
        else:
            logger.info("Barcode image missing or skipped. Sending asset image only.")
            prompt += "\nNote: The second image (Barcode) was not provided or skipped. Please focus only on identifying the asset in Image 1."

        logger.info(f"Sending request to Gemini API (Parts: {len(content_parts)})")
        try:
            response = self.model.generate_content(content_parts)
            logger.debug(f"Raw Gemini response: {response.text}")
        except Exception as api_err:
            logger.error(f"Gemini API call failed: {str(api_err)}")
            raise api_err
        
        # Extract JSON from response
        text = response.text
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                logger.error(f"Could not find JSON in Gemini response: {text}")
                raise ValueError("No JSON found in response")
            json_str = text[start:end]
            data = json.loads(json_str)
            return AuditResult(**data)
        except Exception as e:
            logger.error(f"Failed to parse or validate Gemini JSON: {str(e)}")
            # Fallback for parsing errors
            return AuditResult(
                is_swapped=False,
                match_successful=False,
                mapping_confidence=0.0,
                verification_reason=f"AI Parsing Error: {str(e)}",
                asset_details={"name": "Unknown", "id": None, "description": text[:200], "condition": None},
                barcode_details={"value": None, "position": None, "condition": None}
            )

    def _get_prompt(self, asset_boxes: List[str]) -> str:
        yolo_hint = f"YOLO has detected the following potential regions of interest: {', '.join(asset_boxes)}." if asset_boxes else "YOLO detected no specific regions; please analyze the entire image."
        
        return f"""
        You are a certified physical asset auditor with expert knowledge of industrial, commercial, and office equipment.

You will receive two images for analysis:
- Image 1: Expected to contain a physical asset/object
- Image 2: Expected to contain a barcode/QR code sticker on that same asset

{yolo_hint}
NOTE: These labels are for spatial reference only. DO NOT use these labels as asset names. Identify the asset independently using your own expert vision.

---

STEP 1 — SWAP DETECTION
Examine both images. If Image 1 contains a barcode and Image 2 contains an object, set "is_swapped": true and mentally swap them before proceeding. All analysis below must use the correctly assigned images.

STEP 2 — ASSET IDENTIFICATION
Look at the asset image carefully. Identify the asset using your knowledge of real-world products and equipment.
- "name": Provide the precise, market-accurate product name as it would appear in a product catalog or procurement database. Example: "Herman Miller Aeron Chair" or "Dell UltraSharp 27 Monitor". Never use YOLO detection labels.
- "description": One concise sentence, maximum 20 words. Include asset type, key visible feature, and material or form factor. No filler words.
- "condition": Exactly 10 words or fewer. State the physical condition with a brief visible reason. Format: "[Condition] — [reason]". Example: "Good — no visible scratches or structural damage noted."
- "condition_rating": One of exactly: "Good" | "Damaged" | "Poor"

STEP 3 — BARCODE ANALYSIS
Look at the barcode image carefully.
- "value": Read and transcribe the exact barcode or QR code value. If unreadable, set null.
- "position": Describe where the barcode sticker is located relative to the physical asset. Use directional and spatial language. Example: "Bottom-left corner of the rear panel" or "Center of the top surface near the edge."
- "condition": Exactly 10 words or fewer. State barcode readability with visible reason. Format: "[Condition] — [reason]". Example: "Clear — label intact, no peeling or smudging visible."
- "condition_rating": One of exactly: "Clear" | "Partially Damaged" | "Unreadable"

STEP 4 — MAPPING VERIFICATION
Cross-check both images using:
1. Background color and surface texture consistency
2. Lighting angle and intensity match
3. Environmental surroundings (desk, floor, shelf, etc.)
4. Whether the barcode position described matches what is visible on the asset

If all four checks are consistent → "match_successful": true
If any check fails or is ambiguous → "match_successful": false
Provide your reasoning in "verification_reason" in one clear sentence.

---

STRICT RULES:
- Output ONLY valid raw JSON. No markdown, no code fences, no explanation outside JSON.
- Never hallucinate. If any field cannot be determined with high confidence, set it to null.
- Never use YOLO labels as asset names under any circumstance.
- Never exceed word limits for description and condition fields.
- Temperature is 0.0. Be deterministic and precise.

OUTPUT FORMAT:
{{
  "is_swapped": boolean,
  "match_successful": boolean,
  "mapping_confidence": float between 0.0 and 1.0,
  "verification_reason": string,
  "asset_details": {{
    "name": string | null,
    "description": string | null,
    "condition": string | null,
    "condition_rating": "Good" | "Damaged" | "Poor" | null
  }},
  "barcode_details": {{
    "value": string | null,
    "position": string | null,
    "condition": string | null,
    "condition_rating": "Clear" | "Partially Damaged" | "Unreadable" | null
  }}
}}
        
        """
