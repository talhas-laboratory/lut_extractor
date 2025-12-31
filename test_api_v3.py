
import requests
import json

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def test_payload(label, payload_overrides):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    payload.update(payload_overrides)
        
    print(f"--- Testing: {label} ---")
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")

if __name__ == "__main__":
    # Test 1: reasoning at top level
    test_payload("reasoning: {effort: high}", {"reasoning": {"effort": "high"}})
    
    # Test 2: provider reasoning
    test_payload("provider: {reasoning: {effort: high}}", {"provider": {"reasoning": {"effort": "high"}}})
    
    # Test 3: user's exact phrase in provider
    test_payload("provider: {thinking_level: high}", {"provider": {"thinking_level": "high"}})
