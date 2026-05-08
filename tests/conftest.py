import pytest
from fastapi.testclient import TestClient
from main import app
import os

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_image():
    # Create a small dummy image for testing
    import io
    from PIL import Image
    file = io.BytesIO()
    image = Image.new('RGB', (100, 100))
    image.save(file, 'jpeg')
    file.seek(0)
    return file.read()
