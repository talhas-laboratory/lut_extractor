
import requests
import json
import base64
import numpy as np
from PIL import Image
import io

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def to_b64(arr):
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def test_payload(label, provider_cfg=None, other_cfg=None):
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    ref_b64 = to_b64(img1)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"}
    }
    if provider_cfg:
        payload["provider"] = provider_cfg
    if other_cfg:
        payload.update(other_cfg)
        
    print(f"--- Testing: {label} ---")
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")

if __name__ == "__main__":
    # Test 1: Original (fails)
    test_payload("Original provider: {thinking_level: high}", provider_cfg={"thinking_level": "high"})
    
    # Test 2: provider: {thinking: high}
    test_payload("provider: {thinking: high}", provider_cfg={"thinking": "high"})
    
    # Test 3: provider: {reasoning: high}
    test_payload("provider: {reasoning: high}", provider_cfg={"reasoning": "high"})
    
    # Test 4: include_reasoning: true
    test_payload("include_reasoning: true", other_cfg={"include_reasoning": True})

    # Test 5: No provider, just model
    test_payload("No extra params")
