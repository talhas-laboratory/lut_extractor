
import requests
import json
import base64
import numpy as np
from PIL import Image
import io

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def test_api():
    # Create tiny dummy images
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def to_b64(arr):
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    ref_b64 = to_b64(img1)
    v1_b64 = to_b64(img2)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a color scientist."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze these."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{v1_b64}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "provider": {
            "thinking_level": "high"
        }
    }

    print(f"Testing {OPENROUTER_MODEL}...")
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_api()
