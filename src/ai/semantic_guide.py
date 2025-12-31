"""
Semantic Guide - Enhanced LLM layer to guide the math before histogram matching.

Analyzes source and reference to provide semantic parameters:
- Dominant hue (e.g., "cyan-green" with exact Lab a,b values)
- Hue rotation angle
- Color isolation zones
- Target zone tints
- Accent colors to preserve
"""

import requests
import json
import base64
import io
import numpy as np
from PIL import Image
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"  # Fast model for pre-analysis
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def image_to_base64(img_array: np.ndarray, max_size: int = 512) -> str:
    """Convert numpy image (0-1 float) to base64."""
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    pil_img = Image.fromarray(img_uint8)
    pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=80)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


class SemanticGuide:
    """
    Pre-analysis LLM that provides semantic hints before the math runs.
    
    ENHANCED: Now provides detailed color analysis including:
    - Dominant hue with Lab a,b values
    - Per-zone tinting directions
    - Accent color preservation targets
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE Semantic Guide"
        }
    
    def analyze_pair(self, source_img: np.ndarray, reference_img: np.ndarray) -> Dict:
        """
        Analyze source and reference to suggest semantic parameters.
        
        Returns dict with:
        - luma_strength: 0.0-1.0 (how aggressively to match contrast)
        - preserve_skin: bool (prioritize skin tone accuracy)
        - color_direction: "warm", "cool", or "neutral"
        - shadow_treatment: "crush", "lift", or "preserve"
        - highlight_treatment: "roll", "clip", or "preserve"
        - description: brief explanation of the look
        
        ENHANCED fields:
        - dominant_hue_name: Human-readable (e.g., "cyan-green", "teal", "orange")
        - dominant_hue_lab_a: Target a-channel value for midtones (-128 to 127)
        - dominant_hue_lab_b: Target b-channel value for midtones (-128 to 127)
        - zone_tints: {shadow: {a, b}, midtone: {a, b}, highlight: {a, b}}
        - accent_colors: List of colors to preserve/enhance
        - saturation_hint: 0.8-1.5 multiplier
        """
        src_b64 = image_to_base64(source_img)
        ref_b64 = image_to_base64(reference_img)
        
        prompt = """Analyze these two images for color grading:
1. SOURCE (first): The raw/log footage to be graded
2. REFERENCE (second): The target cinematic look we want to achieve

You are a Senior Colorist. Analyze the REFERENCE image's color grade in detail.

Return a JSON object with these fields:
{
  "luma_strength": <0.0-1.0, how aggressively to match contrast. 0.3=subtle, 0.7=strong>,
  "preserve_skin": <true if reference has natural-looking skin tones>,
  "color_direction": "<warm|cool|neutral>",
  "shadow_treatment": "<crush|lift|preserve>",
  "highlight_treatment": "<roll|clip|preserve>",
  
  "dominant_hue_name": "<name of the dominant color cast, e.g. 'cyan-green', 'teal', 'orange-teal', 'neutral'>",
  "dominant_hue_lab_a": <Lab a-channel target for midtones, -128 to 127. Negative=green, Positive=magenta>,
  "dominant_hue_lab_b": <Lab b-channel target for midtones, -128 to 127. Negative=blue, Positive=yellow>,
  
  "zone_tints": {
    "shadow": {"a": <value>, "b": <value>},
    "midtone": {"a": <value>, "b": <value>},
    "highlight": {"a": <value>, "b": <value>}
  },
  
  "accent_colors": [
    {"name": "<color name>", "preserve": <true|false>, "boost": <0.8-1.5>}
  ],
  
  "saturation_hint": <0.8-1.5>,
  "description": "<1-2 sentence description of the look>"
}

IMPORTANT GUIDELINES:
- For The Matrix look: a≈-8, b≈-15 (cyan-green tint applied to everything)
- For orange/teal: shadows have negative b, highlights have positive b
- For desaturated looks: saturation_hint < 1.0
- For crushed blacks: luma_strength closer to 1.0

Respond ONLY with valid JSON, no markdown or extra text."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{src_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}}
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.2
        }
        
        try:
            print("[GUIDE] Analyzing images for semantic hints (enhanced)...")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Clean markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            guidance = json.loads(content)
            
            # Log the enhanced analysis
            print(f"[GUIDE] Semantic analysis: {guidance.get('description', 'N/A')}")
            print(f"[GUIDE] Dominant hue: {guidance.get('dominant_hue_name', 'neutral')} (a={guidance.get('dominant_hue_lab_a', 0)}, b={guidance.get('dominant_hue_lab_b', 0)})")
            print(f"[GUIDE] Luma: {guidance.get('luma_strength', 0.6)}, Saturation: {guidance.get('saturation_hint', 1.0)}")
            
            if 'zone_tints' in guidance:
                zt = guidance['zone_tints']
                print(f"[GUIDE] Zone tints - Shadow: a={zt.get('shadow', {}).get('a', 0)}, b={zt.get('shadow', {}).get('b', 0)}")
                print(f"[GUIDE] Zone tints - Midtone: a={zt.get('midtone', {}).get('a', 0)}, b={zt.get('midtone', {}).get('b', 0)}")
            
            return guidance
            
            return guidance
            
        except Exception as e:
            print(f"[GUIDE] Analysis failed: {e}, using enhanced defaults")
            return self._get_defaults()

    def analyze_composition(self, reference_img: np.ndarray) -> Dict:
        """
        Deconstruct reference image into semantic color regions.
        
        Returns a "codebook" of compositional elements for precise mapping.
        """
        ref_b64 = image_to_base64(reference_img)
        
        prompt = """Analyze the color composition of this cinematic reference image.
Deconstruct the image into specific semantic regions (e.g., shadows, skin tones, sky, highlights, midtone walls).

For each region, identify:
1. The semantic role (shadow, midtone, highlight, skin, sky, foliage, etc.)
2. The approximate luminance range (0-100)
3. The characteristic tint (Lab a, b values) - BE PRECISE.
4. A brief description

Also identify the GLOBAL tint applied to the whole image.

Return ONLY valid JSON:
{
  "global_tint": {"a": <int>, "b": <int>, "description": "Global cast description"},
  "elements": [
    {
      "role": "shadow|midtone|highlight|skin|sky|foliage",
      "luma_min": <0-100>,
      "luma_max": <0-100>,
      "target_a": <int -128 to 127>,
      "target_b": <int -128 to 127>,
      "description": "..."
    }
  ]
}

IMPORTANT COLOR GUIDELINES:
- Matrix Green: Is NOT cyan. It is OLIVE. a should be negative (-10 to -20), b should be POSITIVE (5 to 15).
- Teal/Orange: Shadows teal (neg a, neg b), Highlights orange (pos a, pos b).
- Sepia: Pos b (yellow), small pos a (red).
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}}
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.2
        }
        
        try:
            print("[GUIDE] Analyzing composition for codebook extraction...")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            data = json.loads(content)
            print(f"[GUIDE] Composition extracted: {len(data.get('elements', []))} elements")
            print(f"[GUIDE] Global tint detected: {data.get('global_tint', {})}")
            return data
            
        except Exception as e:
            print(f"[GUIDE] Composition analysis failed: {e}")
            return {
                "global_tint": {"a": 0, "b": 0, "description": "Neutral fallback"},
                "elements": []
            }
    
    def _get_defaults(self) -> Dict:
        """Return enhanced default values."""
        return {
            "luma_strength": 0.6,
            "preserve_skin": True,
            "color_direction": "neutral",
            "shadow_treatment": "preserve",
            "highlight_treatment": "preserve",
            "dominant_hue_name": "neutral",
            "dominant_hue_lab_a": 0,
            "dominant_hue_lab_b": 0,
            "zone_tints": {
                "shadow": {"a": 0, "b": 0},
                "midtone": {"a": 0, "b": 0},
                "highlight": {"a": 0, "b": 0}
            },
            "accent_colors": [],
            "saturation_hint": 1.2,
            "description": "Default neutral grade"
        }
    
    def merge_with_palette_analysis(self, semantic_hints: Dict, palette_analysis: Dict) -> Dict:
        """
        Combine LLM semantic hints with mathematical palette analysis.
        
        LLM provides high-level intent, palette analysis provides exact values.
        We use LLM to validate/adjust the math, not replace it.
        """
        merged = semantic_hints.copy()
        
        # If LLM detected a strong color direction, validate against palette
        if 'zone_tints' in palette_analysis:
            math_tints = palette_analysis['zone_tints']
            llm_tints = semantic_hints.get('zone_tints', {})
            
            # Average between LLM suggestions and math extraction
            # LLM might know "this is Matrix green" but math knows exact values
            merged_tints = {}
            for zone in ['shadow', 'midtone', 'highlight']:
                math_zone = math_tints.get(zone, {'a': 0, 'b': 0})
                llm_zone = llm_tints.get(zone, {'a': 0, 'b': 0})
                
                # Weight math more heavily (0.7) since it's measured
                merged_tints[zone] = {
                    'a': 0.7 * math_zone.get('a', 0) + 0.3 * llm_zone.get('a', 0),
                    'b': 0.7 * math_zone.get('b', 0) + 0.3 * llm_zone.get('b', 0)
                }
            
            merged['zone_tints'] = merged_tints
        
        # Use palette's neutral shift as ground truth
        if 'neutral_shift' in palette_analysis:
            merged['dominant_hue_lab_a'] = palette_analysis['neutral_shift']['a']
            merged['dominant_hue_lab_b'] = palette_analysis['neutral_shift']['b']
        
        return merged
