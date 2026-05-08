import base64
import io

from PIL import Image


def process_image_for_vlm(image_bytes: bytes):
    # For Gemini, we can send PIL images or base64
    image = Image.open(io.BytesIO(image_bytes))
    return image

def encode_image_to_base64(image_bytes: bytes):
    return base64.b64encode(image_bytes).decode('utf-8')
