"""
Shared test fixtures for the AssetCues Backend test suite.
All fixtures here are available automatically to every test file.
"""
import io
import os

import pytest

# --- Environment Setup ---
# These must be set BEFORE any app imports so that pydantic-settings
# reads them instead of requiring a real .env file.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("VLM_MODEL", "gemini-test")
os.environ.setdefault("YOLO_MODEL", "yolov8n.pt")
os.environ.setdefault("STORAGE_PROVIDER", "local")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

from fastapi.testclient import TestClient

from app.models.schemas import (
    AssetDetails,
    AuditResult,
    BarcodeDetails,
    BoundingBox,
)
from main import app


# ──────────────────────────────────────────────
# Client Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def client():
    """FastAPI TestClient for integration tests."""
    return TestClient(app)


# ──────────────────────────────────────────────
# Image Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def sample_image_bytes():
    """Generate a small 100x100 JPEG image as raw bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(128, 64, 200)).save(buf, "JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_image_large():
    """Generate a larger 640x480 JPEG image for edge-case testing."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(50, 120, 200)).save(buf, "JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def corrupt_image_bytes():
    """Return corrupt bytes that are NOT a valid image."""
    return b"this-is-not-an-image-\x00\xff\xfe"


# ──────────────────────────────────────────────
# Schema Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def mock_audit_result():
    """A valid AuditResult for mocking downstream services."""
    return AuditResult(
        is_swapped=False,
        match_successful=True,
        mapping_confidence=0.95,
        verification_reason="Background and lighting match perfectly.",
        asset_details=AssetDetails(
            name="Dell UltraSharp 27 Monitor",
            id="ASSET-001",
            description="27-inch IPS monitor with thin bezels.",
            condition="Good — no visible scratches.",
            condition_rating="Good",
        ),
        barcode_details=BarcodeDetails(
            value="QR-DELL-27-001",
            position="Bottom-left corner of rear panel.",
            condition="Clear — label intact and legible.",
            condition_rating="Clear",
        ),
    )


@pytest.fixture
def mock_bounding_boxes():
    """Sample YOLO bounding boxes."""
    return [
        BoundingBox(x1=10, y1=20, x2=200, y2=300, confidence=0.92, class_name="tv"),
        BoundingBox(x1=50, y1=60, x2=150, y2=180, confidence=0.45, class_name="remote"),
    ]


# ──────────────────────────────────────────────
# Temporary Storage Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def tmp_storage_path(tmp_path):
    """Provide a temporary directory for local storage adapter tests."""
    return str(tmp_path / "test_audit_sessions")


# ──────────────────────────────────────────────
# Session Metadata Helpers
# ──────────────────────────────────────────────
@pytest.fixture
def valid_session_metadata():
    """Return a valid session metadata dict for one pair (asset + barcode)."""
    return {
        "session_id": "test-session-001",
        "pairs_metadata": [{"pair_index": 0, "barcode_skipped": False}],
    }


@pytest.fixture
def session_metadata_barcode_skipped():
    """Return metadata where the barcode was skipped."""
    return {
        "session_id": "test-session-002",
        "pairs_metadata": [{"pair_index": 0, "barcode_skipped": True}],
    }


@pytest.fixture
def multi_pair_session_metadata():
    """Return metadata for a 3-pair audit session."""
    return {
        "session_id": "test-session-003",
        "pairs_metadata": [
            {"pair_index": 0, "barcode_skipped": False},
            {"pair_index": 1, "barcode_skipped": True},
            {"pair_index": 2, "barcode_skipped": False},
        ],
    }
