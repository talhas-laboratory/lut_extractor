"""
AI Color Scientist - Enhanced with per-zone tinting controls.

Uses OpenRouter API with Gemini to analyze the difference between 
reference and graded images, then suggests specific adjustments to improve the match.

ENHANCED: Now supports per-zone (shadow/midtone/highlight) tinting adjustments.
"""

import os
import json
import requests
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import io
import base64
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def image_to_base64(img_array: np.ndarray, max_size: int = 512) -> str:
    """Convert numpy image array to base64 string for API."""
    pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
    pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


class ColorScientist:
    """
    AI-powered color grade reviewer that compares graded output to reference
    and suggests specific numerical adjustments.
    
    ENHANCED: Now supports per-zone tinting for shadows, midtones, and highlights.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE Color Scientist"
        }
    
    def analyze_and_suggest(
        self, 
        reference_img: np.ndarray, 
        graded_img: np.ndarray,
        current_params: Dict[str, float],
        semantic_context: str = ""
    ) -> Tuple[Dict[str, float], str, bool]:
        """
        Analyze the graded image against reference and suggest adjustments.
        
        ENHANCED: Now with per-zone tinting controls and SEMANTIC CONTEXT.
        """
        ref_b64 = image_to_base64(reference_img)
        graded_b64 = image_to_base64(graded_img)
        
        context_str = ""
        if semantic_context:
            context_str = f"SEMANTIC GOAL: {semantic_context}\n"
        
        prompt = f"""You are a Senior Color Scientist reviewing a color grading result.

{context_str}
CURRENT PARAMETERS:
{json.dumps(current_params, indent=2)}

I'm showing you TWO images:
1. REFERENCE IMAGE (first) - This is the TARGET look we want to achieve
2. GRADED IMAGE (second) - This is our current result after applying color grading

Carefully compare them and provide adjustments.

IMPORTANT ANALYSIS POINTS:
- Does the graded image have the same COLOR TINT as the reference?
- Are the SHADOWS colored correctly (e.g., Matrix has cyan-green shadows)?
- Are the MIDTONES the right hue and saturation?
- Are HIGHLIGHTS matching?
- Is overall saturation and contrast correct?

If specific semantic goals were provided above, ensure the grade matches that feeling.

Respond with a JSON object containing:

1. "adjustments" - numerical changes to apply:
   GLOBAL:
   - "saturation_boost": float (multiply, 0.9 = -10%, 1.1 = +10%)
   - "luma_strength": float (0.0-1.0)
   
   PER-ZONE TINTING (Lab space, -30 to +30):
   - "shadow_tint_a": float (negative = more green, positive = more magenta)
   - "shadow_tint_b": float (negative = more blue, positive = more yellow)
   - "midtone_tint_a": float
   - "midtone_tint_b": float
   - "highlight_tint_a": float
   - "highlight_tint_b": float
   
   LEGACY (still supported):
   - "blue_shift": float (-20 to +20, global b-channel offset)
   - "green_shift": float (-20 to +20, global a-channel offset)
   - "shadow_adjust": float (-15 to +15, darken/lighten shadows)
   - "highlight_adjust": float (-15 to +15, darken/lighten highlights)

2. "feedback" - 1-2 sentence explanation of what you changed and why

3. "satisfied" - boolean, true if the grade looks very close to reference

EXAMPLES:
- For Matrix look: set midtone_tint_a to -8 (green), midtone_tint_b to -12 (cyan)
- For orange/teal: shadow_tint_b to -10 (blue), highlight_tint_b to +10 (yellow)
- For desaturated: reduce saturation_boost below 1.0

Be MORE AGGRESSIVE with adjustments - changes of 5-15 units for tints are appropriate.

Respond ONLY with valid JSON, no markdown code blocks."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{graded_b64}"}
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }

        try:
            print(f"  Calling OpenRouter ({self.model})...")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result_json = response.json()
            response_text = result_json['choices'][0]['message']['content'].strip()
            
            # Clean up response if it has markdown code blocks
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            result = json.loads(response_text)
            
            # Extract components
            adjustments = result.get('adjustments', {})
            feedback = result.get('feedback', 'No feedback provided.')
            satisfied = result.get('satisfied', False)
            
            # Merge adjustments with current params
            new_params = current_params.copy()
            
            # Apply multiplicative adjustments
            if 'saturation_boost' in adjustments:
                new_params['saturation_boost'] = current_params.get('saturation_boost', 1.0) * adjustments['saturation_boost']
                new_params['saturation_boost'] = max(0.5, min(2.0, new_params['saturation_boost']))
            
            if 'luma_strength' in adjustments:
                new_params['luma_strength'] = adjustments['luma_strength']
                new_params['luma_strength'] = max(0.0, min(1.0, new_params['luma_strength']))
            
            # Apply additive adjustments (global color shifts)
            for key in ['blue_shift', 'green_shift', 'shadow_adjust', 'highlight_adjust']:
                if key in adjustments:
                    new_params[key] = current_params.get(key, 0) + adjustments[key]
                    new_params[key] = max(-30, min(30, new_params[key]))
            
            # NEW: Apply per-zone tinting adjustments
            zone_tint_keys = [
                'shadow_tint_a', 'shadow_tint_b',
                'midtone_tint_a', 'midtone_tint_b',
                'highlight_tint_a', 'highlight_tint_b'
            ]
            for key in zone_tint_keys:
                if key in adjustments:
                    new_params[key] = current_params.get(key, 0) + adjustments[key]
                    new_params[key] = max(-40, min(40, new_params[key]))
            
            print(f"  AI Feedback: {feedback}")
            print(f"  Satisfied: {satisfied}")
            print(f"  New params: {json.dumps(new_params, indent=2)}")
            
            return new_params, feedback, satisfied
            
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter API error: {e}")
            return current_params, f"API request failed: {str(e)}", False
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response was: {response_text[:500] if 'response_text' in dir() else 'N/A'}")
            return current_params, f"Failed to parse AI response", False
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            import traceback
            traceback.print_exc()
            return current_params, f"AI analysis failed: {str(e)}", False


def apply_ai_adjustments(
    graded_lab: np.ndarray, 
    params: Dict[str, float]
) -> np.ndarray:
    """
    Apply AI-suggested adjustments to a Lab-space image.
    
    ENHANCED: Now supports per-zone tinting for shadows, midtones, highlights.
    """
    adjusted = graded_lab.copy()
    L = adjusted[:, :, 0]
    
    # =============================================
    # Per-Zone Tinting (NEW)
    # =============================================
    # Define zone weights using smooth transitions
    def zone_weight(L_val, center, width):
        """Gaussian weight centered at L value."""
        return np.exp(-0.5 * ((L_val - center) / width) ** 2)
    
    # Shadow zone (centered at L=15)
    shadow_weight = zone_weight(L, 15, 15)
    # Midtone zone (centered at L=50)
    midtone_weight = zone_weight(L, 50, 20)
    # Highlight zone (centered at L=85)
    highlight_weight = zone_weight(L, 85, 15)
    
    # Apply per-zone tints
    a_shift = np.zeros_like(L)
    b_shift = np.zeros_like(L)
    
    # Shadow tinting
    shadow_a = params.get('shadow_tint_a', 0)
    shadow_b = params.get('shadow_tint_b', 0)
    if shadow_a != 0 or shadow_b != 0:
        a_shift += shadow_weight * shadow_a
        b_shift += shadow_weight * shadow_b
    
    # Midtone tinting
    midtone_a = params.get('midtone_tint_a', 0)
    midtone_b = params.get('midtone_tint_b', 0)
    if midtone_a != 0 or midtone_b != 0:
        a_shift += midtone_weight * midtone_a
        b_shift += midtone_weight * midtone_b
    
    # Highlight tinting
    highlight_a = params.get('highlight_tint_a', 0)
    highlight_b = params.get('highlight_tint_b', 0)
    if highlight_a != 0 or highlight_b != 0:
        a_shift += highlight_weight * highlight_a
        b_shift += highlight_weight * highlight_b
    
    # Apply zone tints
    adjusted[:, :, 1] = adjusted[:, :, 1] + a_shift
    adjusted[:, :, 2] = adjusted[:, :, 2] + b_shift
    
    # =============================================
    # Legacy global adjustments (backward compatible)
    # =============================================
    
    # Apply global green shift (a-channel)
    green_shift = params.get('green_shift', 0)
    if green_shift != 0:
        adjusted[:, :, 1] = adjusted[:, :, 1] + green_shift
    
    # Apply global blue shift (b-channel)
    blue_shift = params.get('blue_shift', 0)
    if blue_shift != 0:
        adjusted[:, :, 2] = adjusted[:, :, 2] + blue_shift
    
    # Apply shadow luminance adjustment
    shadow_adjust = params.get('shadow_adjust', 0)
    if shadow_adjust != 0:
        L = adjusted[:, :, 0]
        shadow_weight_luma = 1 - (L / 100)
        L = L + shadow_adjust * shadow_weight_luma
        adjusted[:, :, 0] = L
    
    # Apply highlight luminance adjustment
    highlight_adjust = params.get('highlight_adjust', 0)
    if highlight_adjust != 0:
        L = adjusted[:, :, 0]
        highlight_weight_luma = L / 100
        L = L + highlight_adjust * highlight_weight_luma
        adjusted[:, :, 0] = L
    
    # Clip to valid Lab ranges
    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0], 0, 100)
    adjusted[:, :, 1] = np.clip(adjusted[:, :, 1], -128, 127)
    adjusted[:, :, 2] = np.clip(adjusted[:, :, 2], -128, 127)
    
    return adjusted
