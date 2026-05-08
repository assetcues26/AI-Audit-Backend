# AI Audit Backend

## Overview

This repository contains the backend API for the `AI-Audit-Backend` project. It is a FastAPI application built to support physical asset auditing using object detection, OCR, and a multimodal large language model workflow.

The backend accepts paired asset and barcode image uploads, runs YOLO detection on the asset image, and then sends the asset and barcode images to a Gemini-based VLM adapter to perform audit verification and asset identification.

## Key Features

- FastAPI-based REST API
- YOLO object detection for asset region extraction
- Gemini VLM integration for asset verification and barcode analysis
- Audit session persistence using local storage adapter
- Structured Pydantic models for consistent responses
- Comprehensive test suite for API validation and business logic

## Repository Structure

- `main.py` - FastAPI application entrypoint
- `app/api/v1/audit.py` - Audit API route and request handling
- `app/core/config.py` - Environment configuration via `pydantic-settings`
- `app/core/logging.py` - Application logging setup
- `app/models/schemas.py` - Pydantic schemas for audit results
- `app/services/audit_engine.py` - Orchestration between YOLO, VLM, and storage
- `app/services/yolo_service.py` - YOLO inference service
- `app/services/vlm_service.py` - VLM service wrapper
- `app/services/vlm_providers/` - Gemini provider implementation and base adapter
- `app/services/storage/` - Local storage adapter for audit session persistence
- `app/utils/image_utils.py` - Image processing helpers for VLM ingestion
- `tests/` - Pytest suite covering API, audit engine, VLM logic, schemas, storage, and utilities
- `audit_sessions/` - Stored session JSON data from completed audits

## Requirements

The backend uses the following core dependencies:

- Python 3.11+
- FastAPI
- Uvicorn
- Ultralytics YOLO
- Google Generative AI client
- Pydantic v2
- Python-dotenv
- Pillow
- NumPy

## Setup

1. Create and activate a Python virtual environment.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Copy the `.env.example` file to `.env` and update the values.

```bash
copy .env.example .env
```

4. Set the required environment variables in `.env`.

```text
GEMINI_API_KEY=your_api_key_here
VLM_MODEL=gemini-1.5-flash
YOLO_MODEL=yolov8n.pt
STORAGE_PROVIDER=local
LOCAL_STORAGE_PATH=./audit_sessions
CORS_ORIGINS=["http://localhost:5173"]
LOG_LEVEL=INFO
```

## Running Locally

Start the API with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The health endpoint is available at:

- `GET /`

The audit endpoint is available at:

- `POST /api/v1/audit/session`

## API Usage

### Create Audit Session

Use multipart form-data to send files and session metadata.

- `images`: List of uploaded images in the order described by metadata
- `session_metadata`: JSON with `session_id` and `pairs_metadata`

Example metadata payload:

```json
{
  "session_id": "12345",
  "pairs_metadata": [
    {"barcode_skipped": false},
    {"barcode_skipped": true}
  ]
}
```

The API processes image pairs and returns a list of `ImageAuditResponse` objects with:

- `pair_index`
- `yolo_boxes`
- `vlm_result`

The VLM result includes an `AuditResult` with asset details, barcode details, match confidence, and verification reasoning.

## Testing

Run the test suite with pytest:

```bash
pytest
```

The repository includes tests for:

- API endpoint validation
- Audit engine orchestration
- VLM adapter behavior
- Schema serialization
- Local storage persistence
- Image utility helpers

## Notes

- The default storage adapter saves audit results to `./audit_sessions`
- `.env` is loaded by `app/core/config.py`; keep API keys secret
- The Gemini adapter expects raw JSON embedded in text responses and performs a best-effort JSON extraction

## Branch and Push

This repository is configured with remote `origin` pointing to:

`https://github.com/assetcues26/AI-Audit-Backend.git`

Use a branch name such as `feature/add-readme`.

```bash
git checkout -b feature/add-readme
git add README.md
git commit -m "Add project README"
git push origin feature/add-readme
```
