from app.services.vlm_providers.gemini_adapter import GeminiAdapter

class VLMService:
    def __init__(self):
        # In a real app, this could be a factory based on settings
        self.provider = GeminiAdapter()

    async def analyze_images(self, asset_image: bytes, barcode_image: bytes, asset_boxes: list):
        return await self.provider.analyze_images(asset_image, barcode_image, asset_boxes)
