"""
ColorDirector - The AI "Director" that controls the mathematical grading engines.

This module implements a 4-stage "Human Simulation" pipeline where the AI acts as
a professional colorist, generating explicit parameters for deterministic math functions.

The AI does NOT generate pixels - it generates PARAMETERS for:
- compute_zoned_luma_mapper (tone curve)
- compute_palette_based_tps (color cast)
- apply_selective_shifts (object corrections)
"""

import requests
import json
import base64
import io
import os  # Added import os
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Fallback valid for testing only if env var is missing (though unlikely to work if expired)
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment")
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
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


@dataclass
class NormalizationParams:
    """Output of analyze_normalization (Tool 1: The Technician)"""
    exposure: float = 0.0       # -1.0 to 1.0 stops
    temperature: float = 0.0    # -100 (cool) to 100 (warm)
    tint: float = 0.0           # -100 (green) to 100 (magenta)
    
    def to_dict(self) -> Dict:
        return {"exposure": self.exposure, "temperature": self.temperature, "tint": self.tint}
    
    def to_log(self) -> List[str]:
        """Generate human-readable log entries."""
        logs = []
        if abs(self.exposure) > 0.1:
            direction = "Boosting" if self.exposure > 0 else "Reducing"
            logs.append(f"{direction} exposure by {self.exposure:+.1f} EV")
        if abs(self.temperature) > 10:
            direction = "Warming" if self.temperature > 0 else "Cooling"
            logs.append(f"{direction} white balance by {abs(self.temperature):.0f}")
        if abs(self.tint) > 10:
            direction = "Adding magenta" if self.tint > 0 else "Adding green"
            logs.append(f"{direction} tint by {abs(self.tint):.0f}")
        return logs if logs else ["No normalization needed"]


@dataclass
class ToneCurveParams:
    """Output of analyze_tone_curve (Tool 2: The Cinematographer)"""
    shadow_method: str = "linear"      # "crush" | "lift" | "linear"
    highlight_method: str = "linear"   # "roll" | "hard" | "linear"
    contrast_volume: float = 1.0       # 0.8 (flat) to 1.5 (punchy)
    
    def to_dict(self) -> Dict:
        return {
            "shadow_method": self.shadow_method,
            "highlight_method": self.highlight_method,
            "contrast_volume": self.contrast_volume
        }
    
    def to_log(self) -> List[str]:
        logs = []
        if self.shadow_method == "crush":
            logs.append("Crushing blacks for cinematic density")
        elif self.shadow_method == "lift":
            logs.append("Lifting shadows for faded/vintage look")
        
        if self.highlight_method == "roll":
            logs.append("Rolling highlights for film-like falloff")
        elif self.highlight_method == "hard":
            logs.append("Hard highlights for digital/punchy look")
        
        if self.contrast_volume < 0.95:
            logs.append(f"Reducing contrast to {self.contrast_volume:.1f}x (flat)")
        elif self.contrast_volume > 1.05:
            logs.append(f"Boosting contrast to {self.contrast_volume:.1f}x (punchy)")
        
        return logs if logs else ["Linear tone curve (neutral)"]


@dataclass
class PaletteIdentityParams:
    """Output of analyze_palette_identity (Tool 3: The Chemist)"""
    shadow_cast: Dict = None       # {"a": float, "b": float}
    highlight_cast: Dict = None    # {"a": float, "b": float}
    neutral_shift: Dict = None     # {"a": float, "b": float} - what gray becomes
    
    def __post_init__(self):
        self.shadow_cast = self.shadow_cast or {"a": 0.0, "b": 0.0}
        self.highlight_cast = self.highlight_cast or {"a": 0.0, "b": 0.0}
        self.neutral_shift = self.neutral_shift or {"a": 0.0, "b": 0.0}
    
    def to_dict(self) -> Dict:
        return {
            "shadow_cast": self.shadow_cast,
            "highlight_cast": self.highlight_cast,
            "neutral_shift": self.neutral_shift
        }
    
    def to_log(self) -> List[str]:
        logs = []
        
        def describe_cast(cast: Dict, zone: str) -> Optional[str]:
            a, b = cast.get("a", 0), cast.get("b", 0)
            if abs(a) < 3 and abs(b) < 3:
                return None
            
            # Determine color name from Lab values
            if a < -5 and b < -5:
                color = "teal"
            elif a < -5 and b > 5:
                color = "olive/green"
            elif a > 5 and b > 5:
                color = "orange/warm"
            elif a > 5 and b < -5:
                color = "magenta/purple"
            elif a < -5:
                color = "green"
            elif a > 5:
                color = "magenta"
            elif b < -5:
                color = "blue"
            elif b > 5:
                color = "yellow"
            else:
                color = "subtle tint"
            
            return f"Detected {color} {zone} cast (a:{a:.1f}, b:{b:.1f})"
        
        for zone, cast in [("shadow", self.shadow_cast), ("highlight", self.highlight_cast)]:
            desc = describe_cast(cast, zone)
            if desc:
                logs.append(desc)
        
        neutral_desc = describe_cast(self.neutral_shift, "neutral/global")
        if neutral_desc:
            logs.append(neutral_desc)
        
        return logs if logs else ["Neutral palette (no color cast)"]


@dataclass 
class SelectiveOperation:
    """A single selective correction operation."""
    region: str           # "skin", "sky", "shadows", "highlights", "foliage"
    action: str           # "protect", "shift_hue", "desaturate", "saturate"
    strength: float = 1.0
    target_a: float = 0.0  # For shift_hue
    target_b: float = 0.0  # For shift_hue
    
    def to_dict(self) -> Dict:
        return {
            "region": self.region,
            "action": self.action,
            "strength": self.strength,
            "target_a": self.target_a,
            "target_b": self.target_b
        }
    
    def to_log(self) -> str:
        if self.action == "protect":
            return f"Protecting {self.region} from color cast (strength: {self.strength:.0%})"
        elif self.action == "shift_hue":
            return f"Shifting {self.region} hue toward (a:{self.target_a:.1f}, b:{self.target_b:.1f})"
        elif self.action == "desaturate":
            return f"Desaturating {self.region} by {self.strength:.0%}"
        elif self.action == "saturate":
            return f"Saturating {self.region} by {self.strength:.0%}"
        return f"Applying {self.action} to {self.region}"


@dataclass
class SelectiveCorrectionParams:
    """Output of analyze_selective_corrections (Tool 4: The Artist)"""
    operations: List[SelectiveOperation] = None
    
    def __post_init__(self):
        self.operations = self.operations or []
    
    def to_dict(self) -> Dict:
        return {"operations": [op.to_dict() for op in self.operations]}
    
    def to_log(self) -> List[str]:
        if not self.operations:
            return ["No selective corrections needed"]
        return [op.to_log() for op in self.operations]


class ColorDirector:
    """
    The AI "Director" that analyzes images and generates parameters for the math engines.
    
    Implements 4 analysis tools matching a professional colorist's workflow:
    1. analyze_normalization - Balance exposure/WB before grading
    2. analyze_tone_curve - Define contrast curve shape
    3. analyze_palette_identity - Extract "Film Stock" identity
    4. analyze_selective_corrections - Identify object-specific tweaks
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE Color Director"
        }
    
    def _call_llm(self, prompt: str, images: List[np.ndarray], max_tokens: int = 4000) -> Dict:
        """Make a call to the LLM with images."""
        image_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img)}"}}
            for img in images
        ]
        
        # Add instruction for Gemini 3 to keep reasoning concise
        full_prompt = prompt + "\n\nIMPORTANT: Keep your reasoning concise. Respond only with valid JSON."
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        *image_content
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        
        try:
            print(f"[DIRECTOR] Calling OpenRouter ({self.model})...")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=90
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                print(f"[DIRECTOR] ERROR: Unexpected response format: {result}")
                return {}
                
            content = result['choices'][0]['message'].get('content', '').strip()
            
            if not content:
                print(f"[DIRECTOR] ERROR: Empty response content. Reasoning likely exhausted tokens. Full JSON: {result}")
                return {}
            
            # Clean markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            return json.loads(content)
            
        except Exception as e:
            print(f"[DIRECTOR] LLM call failed: {e}")
            return {}
    
    # =========================================================================
    # TOOL 1: THE TECHNICIAN - Analyze Normalization
    # =========================================================================
    
    def analyze_normalization(self, src_img: np.ndarray, ref_img: np.ndarray) -> NormalizationParams:
        """
        Analyze exposure and white balance differences between source and reference.
        
        Returns parameters to neutralize the source relative to the reference's
        dynamic range BEFORE applying the color grade.
        """
        prompt = """Analyze these two images for EXPOSURE and WHITE BALANCE differences.

IMAGE 1 (SOURCE): The raw footage to be graded
IMAGE 2 (REFERENCE): The target look we want to achieve

Your task: Determine what normalization the SOURCE needs BEFORE we apply the color grade.
We want to match the SOURCE's exposure and white balance to the REFERENCE's base level.

Return JSON with these exact fields:
{
    "exposure": <float from -1.0 to 1.0, where negative=darken, positive=brighten>,
    "temperature": <float from -100 to 100, where negative=cooler, positive=warmer>,
    "tint": <float from -100 to 100, where negative=more green, positive=more magenta>
}

IMPORTANT:
- Return 0 for any value if no adjustment needed
- This is about matching BASE exposure/WB, not the artistic color grade
- Focus on whether the source is over/underexposed or has wrong WB compared to reference

Respond ONLY with valid JSON, no explanation."""

        print("[DIRECTOR] Tool 1: Analyzing normalization (The Technician)...")
        
        result = self._call_llm(prompt, [src_img, ref_img])
        
        if not result:
            print("[DIRECTOR] Normalization analysis failed, using defaults")
            return NormalizationParams()
        
        params = NormalizationParams(
            exposure=float(result.get("exposure", 0)),
            temperature=float(result.get("temperature", 0)),
            tint=float(result.get("tint", 0))
        )
        
        print(f"[DIRECTOR] Normalization: exp={params.exposure:+.2f}, temp={params.temperature:+.0f}, tint={params.tint:+.0f}")
        return params
    
    # =========================================================================
    # TOOL 2: THE CINEMATOGRAPHER - Analyze Tone Curve
    # =========================================================================
    
    def analyze_tone_curve(self, ref_img: np.ndarray) -> ToneCurveParams:
        """
        Analyze the reference's contrast and tone curve characteristics.
        
        Returns parameters for the luma mapper:
        - shadow_method: How blacks are treated (crush/lift/linear)
        - highlight_method: How highlights are treated (roll/hard/linear)
        - contrast_volume: Overall contrast multiplier
        """
        prompt = """Analyze the CONTRAST and TONE CURVE of this cinematic reference image.

As a cinematographer, determine:
1. How are the SHADOWS treated? (Are blacks crushed/dense, or lifted/faded?)
2. How are the HIGHLIGHTS treated? (Are they rolled off/soft, or hard/digital?)
3. What's the overall CONTRAST level?

Return JSON with these exact fields:
{
    "shadow_method": "<crush|lift|linear>",
    "highlight_method": "<roll|hard|linear>",
    "contrast_volume": <float from 0.8 to 1.5>
}

GUIDELINES:
- "crush": Blacks are very dense, crushed (Matrix, noir, blockbuster)
- "lift": Blacks are lifted, faded (vintage, pastel, Instagram)
- "linear": Natural blacks, no special treatment

- "roll": Highlights compress/rolloff like film (cinematic)
- "hard": Highlights are sharp/digital (modern, HDR)
- "linear": Natural highlights

- contrast_volume: 0.8 = flat/low contrast, 1.0 = normal, 1.5 = very punchy

EXAMPLES:
- Matrix: {"shadow_method": "crush", "highlight_method": "roll", "contrast_volume": 1.2}
- Vintage: {"shadow_method": "lift", "highlight_method": "roll", "contrast_volume": 0.9}
- Blockbuster: {"shadow_method": "crush", "highlight_method": "hard", "contrast_volume": 1.3}

Respond ONLY with valid JSON, no explanation."""

        print("[DIRECTOR] Tool 2: Analyzing tone curve (The Cinematographer)...")
        
        result = self._call_llm(prompt, [ref_img])
        
        if not result:
            print("[DIRECTOR] Tone curve analysis failed, using defaults")
            return ToneCurveParams()
        
        params = ToneCurveParams(
            shadow_method=result.get("shadow_method", "linear"),
            highlight_method=result.get("highlight_method", "linear"),
            contrast_volume=float(result.get("contrast_volume", 1.0))
        )
        
        print(f"[DIRECTOR] Tone: shadows={params.shadow_method}, highlights={params.highlight_method}, contrast={params.contrast_volume:.2f}")
        return params
    
    # =========================================================================
    # TOOL 3: THE CHEMIST - Analyze Palette Identity
    # =========================================================================
    
    def analyze_palette_identity(self, ref_img: np.ndarray) -> PaletteIdentityParams:
        """
        Extract the "Film Stock" identity - the global color cast.
        
        Analyzes what colors the shadows, highlights, and neutral grays become.
        This is the core "look" of the reference.
        """
        prompt = """Analyze the COLOR PALETTE of this cinematic reference image.

As a color chemist, identify the "film stock" identity:
1. What color are the SHADOWS tinted toward?
2. What color are the HIGHLIGHTS tinted toward?
3. What do NEUTRAL GRAYS become? (This is the global cast)

Return JSON with these exact fields (all values in Lab color space):
{
    "shadow_cast": {"a": <float -40 to 40>, "b": <float -40 to 40>},
    "highlight_cast": {"a": <float -40 to 40>, "b": <float -40 to 40>},
    "neutral_shift": {"a": <float -40 to 40>, "b": <float -40 to 40>}
}

LAB COLOR GUIDE:
- a axis: negative = green, positive = magenta/red
- b axis: negative = blue, positive = yellow

EXAMPLES:
- Matrix (olive green): shadows a:-15, b:+8 | highlights a:-5, b:+2
- Orange/Teal: shadows a:-10, b:-15 | highlights a:+10, b:+20
- Sepia: shadows a:+5, b:+15 | highlights a:+3, b:+25
- Noir: shadows a:0, b:0 | highlights a:0, b:0 (neutral)
- Pastel: all values close to 0, very subtle

IMPORTANT: 
- Matrix green is OLIVE (negative a, POSITIVE b), not cyan
- Return 0 for values if the image has no color cast in that zone

Respond ONLY with valid JSON, no explanation."""

        print("[DIRECTOR] Tool 3: Analyzing palette identity (The Chemist)...")
        
        result = self._call_llm(prompt, [ref_img])
        
        if not result:
            print("[DIRECTOR] Palette analysis failed, using defaults")
            return PaletteIdentityParams()
        
        params = PaletteIdentityParams(
            shadow_cast=result.get("shadow_cast", {"a": 0, "b": 0}),
            highlight_cast=result.get("highlight_cast", {"a": 0, "b": 0}),
            neutral_shift=result.get("neutral_shift", {"a": 0, "b": 0})
        )
        
        print(f"[DIRECTOR] Palette: shadows=({params.shadow_cast}), highlights=({params.highlight_cast})")
        return params
    
    # =========================================================================
    # TOOL 4: THE ARTIST - Analyze Selective Corrections
    # =========================================================================
    
    def analyze_selective_corrections(
        self, 
        src_img: np.ndarray, 
        ref_img: np.ndarray
    ) -> SelectiveCorrectionParams:
        """
        Identify specific objects that need independent manipulation.
        
        Compares semantic objects between source and reference to determine
        if specific regions (sky, skin, foliage) need special treatment.
        """
        prompt = """Compare these two images for SELECTIVE COLOR CORRECTIONS needed.

IMAGE 1 (SOURCE): The original footage
IMAGE 2 (REFERENCE): The target color grade

Identify if specific semantic regions need independent adjustments:
- SKIN TONES: Should they be protected from the overall color cast?
- SKY: Does the reference sky have a specific color the source sky lacks?
- FOLIAGE: Are greens treated specially?
- SHADOWS: Should shadow colors be shifted independently?
- HIGHLIGHTS: Should highlight colors be shifted independently?

Return JSON with these exact fields:
{
    "operations": [
        {
            "region": "<skin|sky|foliage|shadows|highlights>",
            "action": "<protect|shift_hue|desaturate|saturate>",
            "strength": <float 0.0 to 1.0>,
            "target_a": <float -40 to 40, for shift_hue only>,
            "target_b": <float -40 to 40, for shift_hue only>
        }
    ]
}

ACTION TYPES:
- "protect": Keep this region closer to its original color (reduce cast effect)
- "shift_hue": Move this region toward specific a,b values
- "desaturate": Reduce color intensity in this region
- "saturate": Boost color intensity in this region

EXAMPLES:
- Matrix with skin: [{"region": "skin", "action": "protect", "strength": 0.7}]
- Teal sky: [{"region": "sky", "action": "shift_hue", "strength": 0.8, "target_a": -15, "target_b": -20}]
- Desaturated shadows: [{"region": "shadows", "action": "desaturate", "strength": 0.5}]

IMPORTANT:
- Only include operations that are ACTUALLY NEEDED
- Return empty operations array if no selective corrections needed
- Maximum 3-4 operations for efficiency

Respond ONLY with valid JSON, no explanation."""

        print("[DIRECTOR] Tool 4: Analyzing selective corrections (The Artist)...")
        
        result = self._call_llm(prompt, [src_img, ref_img])
        
        if not result:
            print("[DIRECTOR] Selective corrections analysis failed, using defaults")
            return SelectiveCorrectionParams()
        
        operations = []
        for op_data in result.get("operations", []):
            operations.append(SelectiveOperation(
                region=op_data.get("region", "unknown"),
                action=op_data.get("action", "protect"),
                strength=float(op_data.get("strength", 1.0)),
                target_a=float(op_data.get("target_a", 0)),
                target_b=float(op_data.get("target_b", 0))
            ))
        
        params = SelectiveCorrectionParams(operations=operations)
        
        print(f"[DIRECTOR] Selective corrections: {len(operations)} operations")
        for op in operations:
            print(f"  - {op.to_log()}")
        
        return params
    
    # =========================================================================
    # ORCHESTRATION - Run full analysis pipeline
    # =========================================================================
    
    def analyze_full(
        self, 
        src_img: np.ndarray, 
        ref_img: np.ndarray
    ) -> Dict:
        """
        Run the complete 4-stage analysis pipeline.
        
        Returns all parameters needed for the math engines plus a human-readable log.
        """
        print("\n" + "="*60)
        print("COLOR DIRECTOR: Running 4-Stage Analysis Pipeline")
        print("="*60)
        
        # Stage 1: Normalization
        norm_params = self.analyze_normalization(src_img, ref_img)
        
        # Stage 2: Tone Curve
        tone_params = self.analyze_tone_curve(ref_img)
        
        # Stage 3: Palette Identity
        palette_params = self.analyze_palette_identity(ref_img)
        
        # Stage 4: Selective Corrections
        selective_params = self.analyze_selective_corrections(src_img, ref_img)
        
        # Build operations log
        operations_log = []
        
        for log in norm_params.to_log():
            operations_log.append({"stage": "normalization", "action": log, "params": norm_params.to_dict()})
        
        for log in tone_params.to_log():
            operations_log.append({"stage": "tone", "action": log, "params": tone_params.to_dict()})
        
        for log in palette_params.to_log():
            operations_log.append({"stage": "palette", "action": log, "params": palette_params.to_dict()})
        
        for log in selective_params.to_log():
            operations_log.append({"stage": "selective", "action": log, "params": selective_params.to_dict()})
        
        print("\n" + "="*60)
        print("COLOR DIRECTOR: Analysis Complete")
        print("="*60 + "\n")
        
        return {
            "normalization": norm_params,
            "tone_curve": tone_params,
            "palette_identity": palette_params,
            "selective_corrections": selective_params,
            "operations_log": operations_log
        }
