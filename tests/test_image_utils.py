"""
Unit Tests — Image Utility Functions
Tests the image processing and base64 encoding helpers.
"""
import io
import base64
import pytest
from PIL import Image
from app.utils.image_utils import process_image_for_vlm, encode_image_to_base64


class TestProcessImageForVLM:
    def test_returns_pil_image(self, sample_image_bytes):
        result = process_image_for_vlm(sample_image_bytes)
        assert isinstance(result, Image.Image)

    def test_image_dimensions_preserved(self, sample_image_bytes):
        result = process_image_for_vlm(sample_image_bytes)
        assert result.size == (100, 100)

    def test_large_image(self, sample_image_large):
        result = process_image_for_vlm(sample_image_large)
        assert result.size == (640, 480)

    def test_corrupt_image_raises(self, corrupt_image_bytes):
        with pytest.raises(Exception):
            process_image_for_vlm(corrupt_image_bytes)


class TestEncodeImageToBase64:
    def test_returns_base64_string(self, sample_image_bytes):
        result = encode_image_to_base64(sample_image_bytes)
        assert isinstance(result, str)

    def test_base64_is_decodable(self, sample_image_bytes):
        encoded = encode_image_to_base64(sample_image_bytes)
        decoded = base64.b64decode(encoded)
        assert decoded == sample_image_bytes

    def test_empty_bytes(self):
        result = encode_image_to_base64(b"")
        assert result == ""
