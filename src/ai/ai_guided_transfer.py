"""
AI-Guided Color Transfer - Semantic Color Mapping via LLM

This module generates a "Color DNA" JSON recipe from an LLM analysis
of source and reference images, then applies the recipe transformations.

Unlike Reinhard (blind statistics), this uses LLM vision to understand:
- Zone-specific tints (shadows vs midtones vs highlights)
- Non-linear curves for cinematic contrast
- Semantic intent (e.g., "Matrix green" = olive midtones + teal shadows)
"""

import cv2
import numpy as np
import requests
import json
import base64
import io
import os
from PIL import Image
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.interpolate import interp1d
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "google/gemini-3-pro-preview"  # Best vision reasoning
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ColorRecipe:
    """The 'Color DNA' - explicit transformation parameters."""
    # Global adjustments
    exposure_mult: float = 1.0
    brightness_offset: float = 0.0
    
    # Channel shifts in LAB space
    a_channel_shift: float = 0.0  # Negative = green, Positive = magenta
    b_channel_shift: float = 0.0  # Negative = blue, Positive = yellow
    
    # Non-linear curves: list of [input, output] points (0-255 scale)
    l_curve: list = None  # Luminance curve
    a_curve: list = None  # a-channel curve
    b_curve: list = None  # b-channel curve
    
    # Saturation
    saturation_mult: float = 1.0
    
    # Zone-specific tints
    shadow_a: float = 0.0
    shadow_b: float = 0.0
    midtone_a: float = 0.0
    midtone_b: float = 0.0
    highlight_a: float = 0.0
    highlight_b: float = 0.0
    
    # Metadata
    description: str = ""
    
    def __post_init__(self):
        # Default identity curves
        if self.l_curve is None:
            self.l_curve = [[0, 0], [128, 128], [255, 255]]
        if self.a_curve is None:
            self.a_curve = [[0, 0], [128, 128], [255, 255]]
        if self.b_curve is None:
            self.b_curve = [[0, 0], [128, 128], [255, 255]]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ColorRecipe':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def image_to_base64(img_array: np.ndarray, max_size: int = 512) -> str:
    """Convert numpy image (0-1 float or 0-255 uint8) to base64 JPEG."""
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        if img_array.max() <= 1.0:
            img_uint8 = (img_array * 255).astype(np.uint8)
        else:
            img_uint8 = img_array.astype(np.uint8)
    else:
        img_uint8 = img_array
    
    pil_img = Image.fromarray(img_uint8)
    pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=80)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def apply_curve(channel: np.ndarray, curve_points: list) -> np.ndarray:
    """
    Apply a non-linear curve transformation to a channel.
    
    Args:
        channel: Single channel array (0-255 scale)
        curve_points: List of [input, output] points, e.g. [[0,0], [128,120], [255,250]]
    
    Returns:
        Transformed channel
    """
    if not curve_points or len(curve_points) < 2:
        return channel
    
    # Extract x and y values
    x_points = [p[0] for p in curve_points]
    y_points = [p[1] for p in curve_points]
    
    # Create interpolation function
    curve_func = interp1d(x_points, y_points, kind='linear', 
                          bounds_error=False, fill_value=(y_points[0], y_points[-1]))
    
    # Apply curve
    return curve_func(channel)


def apply_zone_tints(l_channel: np.ndarray, 
                     a_channel: np.ndarray, 
                     b_channel: np.ndarray,
                     recipe: ColorRecipe) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply zone-specific color tints based on luminance.
    
    Shadows (L < 40), Midtones (40 < L < 70), Highlights (L > 70)
    """
    # Normalize L to 0-100 for zone detection
    l_normalized = l_channel / 2.55  # Convert 0-255 to 0-100
    
    # Zone weights (soft transitions using gaussian-like falloff)
    shadow_weight = np.exp(-0.5 * ((l_normalized - 20) / 20) ** 2)
    midtone_weight = np.exp(-0.5 * ((l_normalized - 50) / 20) ** 2)
    highlight_weight = np.exp(-0.5 * ((l_normalized - 80) / 20) ** 2)
    
    # Normalize weights
    total_weight = shadow_weight + midtone_weight + highlight_weight + 1e-6
    shadow_weight /= total_weight
    midtone_weight /= total_weight
    highlight_weight /= total_weight
    
    # Apply zone-specific tints
    a_tint = (shadow_weight * recipe.shadow_a + 
              midtone_weight * recipe.midtone_a + 
              highlight_weight * recipe.highlight_a)
    
    b_tint = (shadow_weight * recipe.shadow_b + 
              midtone_weight * recipe.midtone_b + 
              highlight_weight * recipe.highlight_b)
    
    return a_channel + a_tint, b_channel + b_tint


def apply_ai_recipe(source_img: np.ndarray, recipe: ColorRecipe) -> np.ndarray:
    """
    Apply the AI-generated Color DNA recipe to an image.
    
    Args:
        source_img: RGB image as float32 (0-1) or uint8 (0-255)
        recipe: ColorRecipe with transformation parameters
    
    Returns:
        Graded RGB image as float32 (0-1)
    """
    # Ensure uint8 for OpenCV
    if source_img.dtype == np.float32 or source_img.dtype == np.float64:
        if source_img.max() <= 1.0:
            src_uint8 = (source_img * 255).astype(np.uint8)
        else:
            src_uint8 = source_img.astype(np.uint8)
    else:
        src_uint8 = source_img
    
    # Convert RGB to BGR for OpenCV, then to LAB
    src_bgr = cv2.cvtColor(src_uint8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # 1. EXPOSURE: Apply global tone mapping
    l = l * recipe.exposure_mult + recipe.brightness_offset
    
    # 2. CURVES: Apply non-linear transformations
    l = apply_curve(l, recipe.l_curve)
    a = apply_curve(a, recipe.a_curve)
    b = apply_curve(b, recipe.b_curve)
    
    # 3. GLOBAL HUE SHIFTS
    a = a + recipe.a_channel_shift
    b = b + recipe.b_channel_shift
    
    # 4. ZONE TINTS: Apply zone-specific color shifts
    a, b = apply_zone_tints(l, a, b, recipe)
    
    # 5. SATURATION: Scale a,b channels around neutral (128)
    a = 128 + (a - 128) * recipe.saturation_mult
    b = 128 + (b - 128) * recipe.saturation_mult
    
    # Clip to valid LAB ranges
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    # Merge and convert back
    merged = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    result_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    # Return as float32 (0-1)
    return result_rgb.astype(np.float32) / 255.0


class AIGuidedTransfer:
    """
    Main class for AI-guided color transfer.
    
    Workflow:
    1. generate_recipe() - LLM analyzes images, returns ColorRecipe
    2. apply_recipe() - Apply recipe to source image
    3. refine_recipe() - User feedback updates recipe via LLM
    4. bake_to_hald() - Apply recipe to neutral HALD for LUT export
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE AI-Guided Transfer"
        }
    
    def generate_recipe(self, source_img: np.ndarray, reference_img: np.ndarray) -> ColorRecipe:
        """
        Send images to LLM and generate a Color DNA recipe.
        
        The LLM analyzes the reference's color grade and generates explicit
        transformation parameters to apply to the source.
        """
        src_b64 = image_to_base64(source_img)
        ref_b64 = image_to_base64(reference_img)
        
        prompt = """You are a Professional Film Colorist. Analyze these two images:

1. SOURCE (first image): The raw footage to be color graded
2. REFERENCE (second image): The target cinematic look to replicate

Generate a precise COLOR RECIPE as JSON to transform the SOURCE to match the REFERENCE's look.

Return ONLY valid JSON with these exact fields:
{
  "exposure_mult": <0.8-1.2, global exposure multiplier>,
  "brightness_offset": <-20 to 20, L channel offset>,
  
  "a_channel_shift": <-30 to 30, global green/magenta shift. Negative=green, Positive=magenta>,
  "b_channel_shift": <-30 to 30, global blue/yellow shift. Negative=blue, Positive=yellow>,
  
  "l_curve": [[0, <0-10>], [128, <100-150>], [255, <245-255>]],
  "a_curve": [[0, <0-20>], [128, <110-145>], [255, <235-255>]],
  "b_curve": [[0, <0-20>], [128, <110-145>], [255, <235-255>]],
  
  "saturation_mult": <0.7-1.3>,
  
  "shadow_a": <-20 to 20, shadow green/magenta tint>,
  "shadow_b": <-20 to 20, shadow blue/yellow tint>,
  "midtone_a": <-20 to 20>,
  "midtone_b": <-20 to 20>,
  "highlight_a": <-10 to 10>,
  "highlight_b": <-10 to 10>,
  
  "description": "<Brief description of the look>"
}

CRITICAL COLOR GUIDELINES:
- Matrix Green: Olive/sickly green. a_channel_shift ≈ -15, midtone_a ≈ -20, shadow_b ≈ -10 (teal shadows)
- Orange/Teal: shadow_b negative (teal), highlight_b positive (orange)
- Crushed blacks: l_curve[0][1] > 0 (lifts black point)
- Faded look: Lower saturation_mult, gentler curves

Respond ONLY with the JSON object, no explanation."""

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
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        try:
            print("[AI-RECIPE] Generating Color DNA from LLM...")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message'].get('content', '').strip()
            
            if not content:
                print("[AI-RECIPE] ERROR: Empty response")
                return ColorRecipe(description="Fallback: LLM returned empty")
            
            # Clean markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])
            
            recipe_dict = json.loads(content)
            recipe = ColorRecipe.from_dict(recipe_dict)
            
            print(f"[AI-RECIPE] Generated: {recipe.description}")
            print(f"[AI-RECIPE] Global shifts: a={recipe.a_channel_shift}, b={recipe.b_channel_shift}")
            print(f"[AI-RECIPE] Zone tints - Shadow: a={recipe.shadow_a}, b={recipe.shadow_b}")
            
            return recipe
            
        except json.JSONDecodeError as e:
            print(f"[AI-RECIPE] JSON parse error: {e}")
            print(f"[AI-RECIPE] Raw content: {content[:500]}")
            return ColorRecipe(description="Fallback: JSON parse failed")
        except Exception as e:
            print(f"[AI-RECIPE] Error: {e}")
            return ColorRecipe(description=f"Fallback: {str(e)}")
    
    def refine_recipe(self, current_recipe: ColorRecipe, 
                      reference_img: np.ndarray,
                      graded_img: np.ndarray,
                      user_feedback: str) -> ColorRecipe:
        """
        Refine the recipe based on user feedback.
        
        User can say things like:
        - "More green"
        - "Desaturate the shadows"
        - "Make highlights warmer"
        """
        ref_b64 = image_to_base64(reference_img)
        graded_b64 = image_to_base64(graded_img)
        
        prompt = f"""You are refining a color grade based on user feedback.

CURRENT RECIPE:
{json.dumps(current_recipe.to_dict(), indent=2)}

USER FEEDBACK: "{user_feedback}"

Look at the REFERENCE (first image) and CURRENT GRADED RESULT (second image).
Adjust the recipe to address the user's feedback while staying true to the reference look.

Return the COMPLETE updated recipe as JSON (same format as before).
Only modify the values that need to change based on the feedback.
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{graded_b64}"}}
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        try:
            print(f"[AI-RECIPE] Refining based on: '{user_feedback}'")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message'].get('content', '').strip()
            
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])
            
            recipe_dict = json.loads(content)
            recipe = ColorRecipe.from_dict(recipe_dict)
            
            print(f"[AI-RECIPE] Refined: {recipe.description}")
            return recipe
            
        except Exception as e:
            print(f"[AI-RECIPE] Refinement failed: {e}")
            return current_recipe  # Return unchanged on error
    
    def apply_recipe(self, source_img: np.ndarray, recipe: ColorRecipe) -> np.ndarray:
        """Apply a ColorRecipe to an image."""
        return apply_ai_recipe(source_img, recipe)
    
    def bake_to_hald(self, recipe: ColorRecipe, size: int = 33) -> np.ndarray:
        """
        Apply the recipe to a neutral HALD lattice to generate a LUT.
        
        Returns the graded lattice ready for .cube export.
        """
        from src.io.baker import generate_identity_lattice
        
        # Generate neutral HALD (N^3 x 3 RGB values)
        hald = generate_identity_lattice(size=size)
        
        # Reshape to image-like format for processing
        # Each "row" is a slice of the color cube
        hald_img = hald.reshape(size, size * size, 3)
        
        # Apply recipe
        graded_img = self.apply_recipe(hald_img, recipe)
        
        # Reshape back to lattice
        graded_lattice = graded_img.reshape(-1, 3)
        
        return graded_lattice
