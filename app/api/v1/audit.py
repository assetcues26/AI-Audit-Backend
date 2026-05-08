import json
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ImageAuditResponse
from app.services.audit_engine import AuditEngine

router = APIRouter()

@router.post("/session", response_model=List[ImageAuditResponse])
async def create_audit_session(
    images: List[UploadFile] = File(...),
    session_metadata: str = Form(...)
):
    try:
        metadata = json.loads(session_metadata)
        session_id = metadata["session_id"]
        pairs_meta = metadata["pairs_metadata"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session metadata")

    # Reconstruct pairs from files based on metadata.
    # Metadata tells us which file is asset and which is barcode for each pair.
    # For simplicity in this POC, we assume files are sent in pairs (Asset, Barcode)
    # in the order matching pairs_metadata.

    image_contents = []
    for img in images:
        content = await img.read()
        image_contents.append(content)

    pairs = []
    file_ptr = 0
    for meta in pairs_meta:
        if file_ptr < len(image_contents):
            asset_img = image_contents[file_ptr]
            file_ptr += 1
        else:
            raise HTTPException(status_code=400, detail="Missing asset image for pair")

        barcode_img = b""
        if not meta["barcode_skipped"]:
            if file_ptr < len(image_contents):
                barcode_img = image_contents[file_ptr]
                file_ptr += 1
            else:
                raise HTTPException(status_code=400, detail="Missing barcode image for pair")

        pairs.append({"asset": asset_img, "barcode": barcode_img})

    engine = AuditEngine()
    results = await engine.process_batch(session_id, pairs)
    return results
