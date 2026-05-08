import logging

from app.models.schemas import AuditResult, ImageAuditResponse
from app.services.storage.local_adapter import LocalStorageAdapter
from app.services.vlm_service import VLMService
from app.services.yolo_service import YOLOService

logger = logging.getLogger(__name__)

class AuditEngine:
    def __init__(self):
        self.yolo = YOLOService()
        self.vlm = VLMService()
        self.storage = LocalStorageAdapter()

    async def process_batch(self, session_id: str, pairs: list):
        logger.info(f"Starting audit session {session_id} with {len(pairs)} pairs")
        results = []
        for i, pair in enumerate(pairs):
            try:
                logger.info(f"Processing pair {i+1}/{len(pairs)}")
                asset_image = pair['asset']
                barcode_image = pair['barcode']

                # 1. Run YOLO on asset image
                logger.debug(f"Running YOLO on asset {i}")
                yolo_boxes = self.yolo.detect_objects(asset_image)
                box_names = [b.class_name for b in yolo_boxes]
                logger.info(f"YOLO detected: {box_names}")

                # 2. Run VLM
                logger.info(f"Dispatching to VLM for pair {i}")
                vlm_result = await self.vlm.analyze_images(asset_image, barcode_image, box_names)
                logger.info(f"VLM analysis complete for pair {i}. Match: {vlm_result.match_successful}")

                results.append(ImageAuditResponse(
                    pair_index=i,
                    yolo_boxes=yolo_boxes,
                    vlm_result=vlm_result
                ))
            except Exception as e:
                logger.error(f"Critical error processing pair {i}: {str(e)}", exc_info=True)
                results.append(ImageAuditResponse(
                    pair_index=i,
                    yolo_boxes=[],
                    vlm_result=AuditResult(
                        is_swapped=False,
                        match_successful=False,
                        mapping_confidence=0.0,
                        verification_reason=f"Processing error: {str(e)}",
                        asset_details={"name": "Error", "id": None, "description": f"Failed at step: {str(e)}", "condition": None},
                        barcode_details={"value": None, "position": None, "condition": None}
                    )
                ))

        # Save results
        logger.info(f"Saving session {session_id} to storage")
        session_data = [r.model_dump() for r in results]
        self.storage.save_session(session_id, session_data)

        return results
