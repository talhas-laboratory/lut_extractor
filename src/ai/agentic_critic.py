"""
Agentic Critic Layer - Senior Colorist AI that optimizes TPS Pin Coordinates.
Uses Gemini 3 Pro with thinking_level="high" to minimize perceptual differences.
"""

import os
import json
import requests
import numpy as np
import base64
import io
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Using gemini-2.0-flash-thinking-preview or similar for high reasoning if gemini-3-pro isn't available yet in OpenRouter
# But following user's spec: gemini-3-pro-preview
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
# However, the user explicitly asked for gemini-3-pro-preview. I will use that identifier.
OPENROUTER_MODEL_SPEC = "google/gemini-3-pro-preview" 
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def image_to_base64(img_array: np.ndarray, max_size: int = 768) -> str:
    """Convert numpy image array (0-1 float) to base64 string for API."""
    # Handle image format
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
        
    pil_img = Image.fromarray(img_uint8)
    pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')

class AgenticCritic:
    """
    The Senior Colorist AI. Analyzes the delta between V1 Result and Reference,
    then rewrites the TPS Pin JSON to fix errors.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL_SPEC
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE Agentic Critic"
        }

    def critique_and_optimize(
        self, 
        reference_img: np.ndarray, 
        v1_img: np.ndarray, 
        pins_json: List[Dict]
    ) -> Tuple[List[Dict], str, bool]:
        """
        Send images and pins to Gemini 3 Pro for optimization.
        """
        ref_b64 = image_to_base64(reference_img)
        v1_b64 = image_to_base64(v1_img)

        system_instruction = (
            "You are a Senior Color Scientist. Your goal is to minimize the perceptual difference "
            "between a Reference Movie Still and a Graded Source. You will be provided with the "
            "current mathematical coordinates (TPS Pins) of the color transform in RGB space [0, 1]. "
            "You must return a modified JSON of these coordinates to correct visual errors like "
            "skin-tone health and shadow density."
        )

        user_prompt = (
            "I am providing three items:\n"
            "1. THE GOAL: The original Movie Reference image.\n"
            "2. THE RESULT: The V1 Graded Frame (our best mathematical guess).\n"
            "3. THE KNOBS: A JSON list of TPS Pin Coordinates currently being used.\n\n"
            "CURRENT PINS JSON:\n"
            f"{json.dumps(pins_json, indent=2)}\n\n"
            "TASK:\n"
            "1. Analyze the visual delta. Are shadows too green? Is skin too pale? Is the overall contrast correct?\n"
            "2. Identify which Pins in the JSON correspond to the problematic color regions.\n"
            "3. Rewrite the JSON by modifying the 'r', 'g', 'b' values of specific pins to pull the colors closer to the reference.\n"
            f"4. Return a JSON object with EXACTLY {len(pins_json)} pins in the 'optimized_pins' array (same count as input), and a brief 'feedback' field.\n"
            "\n"
            "CRITICAL: You MUST return ALL pins from the input, not just the ones you modified. Keep the same 'id' and 'label' for each pin."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{v1_b64}"}
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "reasoning": {
                "effort": "high"
            },
            "include_reasoning": True,
            "max_tokens": 4000
        }


        try:
            print(f"[CRITIC] Calling {self.model} with reasoning.effort='high'...", flush=True)
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=120
            )

            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                print(f"[CRITIC] ERROR: Unexpected response format: {result}")
                return pins_json, "API returned unexpected format", False
                
            content = result['choices'][0]['message'].get('content', '').strip()
            
            if not content:
                print(f"[CRITIC] ERROR: Empty response content. Reasoning likely exhausted tokens. Full JSON: {result}")
                return pins_json, "AI returned empty response (token limit?)", False
            
            # Parse the JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[CRITIC] JSON parse error: {e}")
                print(f"[CRITIC] Response was: {content}")
                return pins_json, "Failed to parse AI optimization", False
                
            optimized_pins = data.get('optimized_pins', pins_json)
            feedback = data.get('feedback', "Optimization complete.")
            satisfied = data.get('satisfied', False) # Optional from AI
            
            print(f"[CRITIC] AI Feedback: {feedback}")
            return optimized_pins, feedback, satisfied

        except Exception as e:
            print(f"[CRITIC] Error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"[CRITIC] Response Body: {response.text}")
            return pins_json, f"Critic failed: {str(e)}", False


