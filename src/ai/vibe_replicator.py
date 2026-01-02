"""
Vibe Replicator - Agentic LLM Colorist Workflow

This module implements a 5-phase agentic loop that analyzes a reference image
like a professional colorist, understands its vibe and color grading intent,
generates replication instructions, applies them, and self-critiques the result.

The agentic loop:
1. analyze_vibe() - Understand reference's mood/emotion/color intent
2. generate_instructions() - Create colorist-level instructions (Lab values, curves)
3. apply_instructions() - Apply the grade to the source image
4. critique() - Compare result to reference, score vibe match
5. refine_instructions() - If not satisfied, adjust and re-apply

Uses Gemini 3 Pro via OpenRouter API.
"""

import requests
import json
import base64
import io
import os
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
import cv2

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment")
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def image_to_base64(img_array: np.ndarray, max_size: int = 512) -> str:
    """Convert numpy image (0-1 float or 0-255 uint8) to base64 JPEG."""
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


# =============================================================================
# DATA CLASSES - Structured outputs from each agentic phase
# =============================================================================

@dataclass
class VibeAnalysis:
    """Output of Phase 1: Understanding the reference's vibe and grading."""
    vibe_description: str = ""           # "Film noir tension", "Warm nostalgic summer"
    mood_keywords: List[str] = field(default_factory=list)  # ["tense", "mysterious", "dark"]
    color_grading_style: str = ""        # "Crushed blacks with teal shadows"
    technical_observations: Dict = field(default_factory=dict)  # Lab values, contrast notes
    
    def to_dict(self) -> Dict:
        return {
            "vibe_description": self.vibe_description,
            "mood_keywords": self.mood_keywords,
            "color_grading_style": self.color_grading_style,
            "technical_observations": self.technical_observations
        }


@dataclass
class GradingInstructions:
    """Output of Phase 2: Professional colorist instructions."""
    # Exposure and contrast
    exposure_adjustment: float = 0.0     # -1.0 to 1.0 EV
    contrast_curve: str = "linear"       # "s-curve", "linear", "lifted-blacks", "crushed"
    contrast_strength: float = 1.0       # 0.8 to 1.5
    
    # Zone-based color (Lab a,b values)
    shadow_a: float = 0.0
    shadow_b: float = 0.0
    midtone_a: float = 0.0
    midtone_b: float = 0.0
    highlight_a: float = 0.0
    highlight_b: float = 0.0
    
    # Saturation
    saturation_mult: float = 1.0         # 0.5 to 1.5
    
    # Special notes
    special_notes: List[str] = field(default_factory=list)  # ["Protect skin tones"]
    
    # Description from AI
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "exposure_adjustment": self.exposure_adjustment,
            "contrast_curve": self.contrast_curve,
            "contrast_strength": self.contrast_strength,
            "shadow_a": self.shadow_a,
            "shadow_b": self.shadow_b,
            "midtone_a": self.midtone_a,
            "midtone_b": self.midtone_b,
            "highlight_a": self.highlight_a,
            "highlight_b": self.highlight_b,
            "saturation_mult": self.saturation_mult,
            "special_notes": self.special_notes,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GradingInstructions':
        return cls(
            exposure_adjustment=float(data.get("exposure_adjustment", 0)),
            contrast_curve=data.get("contrast_curve", "linear"),
            contrast_strength=float(data.get("contrast_strength", 1.0)),
            shadow_a=float(data.get("shadow_a", 0)),
            shadow_b=float(data.get("shadow_b", 0)),
            midtone_a=float(data.get("midtone_a", 0)),
            midtone_b=float(data.get("midtone_b", 0)),
            highlight_a=float(data.get("highlight_a", 0)),
            highlight_b=float(data.get("highlight_b", 0)),
            saturation_mult=float(data.get("saturation_mult", 1.0)),
            special_notes=data.get("special_notes", []),
            description=data.get("description", "")
        )


@dataclass
class CritiqueResult:
    """Output of Phase 4: Self-assessment."""
    vibe_match_score: int = 5            # 1-10
    issues_found: List[str] = field(default_factory=list)  # ["Shadows too blue"]
    suggested_adjustments: Dict = field(default_factory=dict)  # Parameter changes
    satisfied: bool = False              # If true, no more iterations
    feedback: str = ""                   # Human-readable critique
    analysis_of_edits: str = ""          # NEW: Reverse-engineered analysis of what was done
    
    def to_dict(self) -> Dict:
        return {
            "vibe_match_score": self.vibe_match_score,
            "issues_found": self.issues_found,
            "suggested_adjustments": self.suggested_adjustments,
            "satisfied": self.satisfied,
            "feedback": self.feedback,
            "analysis_of_edits": self.analysis_of_edits
        }


# =============================================================================
# GRADING FUNCTIONS - Apply the instructions to images
# =============================================================================

def apply_contrast_curve(l_channel: np.ndarray, curve_type: str, strength: float) -> np.ndarray:
    """Apply contrast curve transformation to L channel."""
    # Normalize to 0-1
    l_norm = l_channel / 100.0
    
    if curve_type == "s-curve":
        # S-curve: sigmoid-like transformation
        # Increase contrast in midtones, compress shadows/highlights
        mid = 0.5
        l_norm = mid + (l_norm - mid) * strength
        # Apply slight S shape
        l_norm = np.clip(l_norm, 0, 1)
        l_norm = l_norm ** (1.0 / (0.8 + 0.4 * strength))  # Gamma adjustment
        
    elif curve_type == "lifted-blacks":
        # Lift shadows, vintage/faded look
        lift = 0.08 * strength
        l_norm = l_norm * (1 - lift) + lift
        
    elif curve_type == "crushed":
        # Crush blacks, cinematic density
        crush = 0.1 * strength
        l_norm = np.maximum(l_norm - crush, 0) / (1 - crush)
        l_norm = l_norm * (1 + 0.1 * strength)  # Boost overall contrast
        
    # Linear: no change
    
    return np.clip(l_norm * 100, 0, 100)


def apply_zone_tints(
    l_channel: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    instructions: GradingInstructions
) -> tuple:
    """Apply zone-specific color tints based on luminance."""
    
    def zone_weight(L_val, center, width):
        """Gaussian weight centered at L value."""
        return np.exp(-0.5 * ((L_val - center) / width) ** 2)
    
    # Zone masks (soft, overlapping)
    shadow_weight = zone_weight(l_channel, 25, 20)    # Centered at L=25
    midtone_weight = zone_weight(l_channel, 50, 25)   # Centered at L=50
    highlight_weight = zone_weight(l_channel, 80, 20) # Centered at L=80
    
    # Apply tints
    a_out = a_channel.copy()
    b_out = b_channel.copy()
    
    a_out += shadow_weight * instructions.shadow_a
    b_out += shadow_weight * instructions.shadow_b
    
    a_out += midtone_weight * instructions.midtone_a
    b_out += midtone_weight * instructions.midtone_b
    
    a_out += highlight_weight * instructions.highlight_a
    b_out += highlight_weight * instructions.highlight_b
    
    # Clamp to valid Lab range
    a_out = np.clip(a_out, -128, 127)
    b_out = np.clip(b_out, -128, 127)
    
    return a_out, b_out


def apply_grading_instructions(source_img: np.ndarray, instructions: GradingInstructions) -> np.ndarray:
    """
    Apply the AI-generated grading instructions to a source image.
    
    Args:
        source_img: RGB image as float32 (0-1) or uint8 (0-255)
        instructions: GradingInstructions with transformation parameters
    
    Returns:
        Graded RGB image as float32 (0-1)
    """
    # Ensure float32 0-1
    if source_img.max() > 1.0:
        img = source_img.astype(np.float32) / 255.0
    else:
        img = source_img.astype(np.float32)
    
    # Convert to Lab
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    L = lab[:, :, 0]  # 0-255 in OpenCV Lab
    a = lab[:, :, 1] - 128  # Center at 0
    b = lab[:, :, 2] - 128  # Center at 0
    
    # Convert L to standard 0-100 range for processing
    L = L / 255.0 * 100.0
    
    # 1. Apply exposure adjustment
    if instructions.exposure_adjustment != 0:
        # EV adjustment: multiply by 2^EV
        ev_mult = 2 ** instructions.exposure_adjustment
        L = L * ev_mult
    
    # 2. Apply contrast curve
    L = apply_contrast_curve(L, instructions.contrast_curve, instructions.contrast_strength)
    
    # 3. Apply zone-based color tints
    a, b = apply_zone_tints(L, a, b, instructions)
    
    # 4. Apply saturation multiplier
    chroma = np.sqrt(a**2 + b**2)
    sat_factor = instructions.saturation_mult
    # Scale a,b by saturation factor
    a = a * sat_factor
    b = b * sat_factor
    
    # Clamp final values
    L = np.clip(L, 0, 100)
    a = np.clip(a, -128, 127)
    b = np.clip(b, -128, 127)
    
    # Convert back to OpenCV Lab format
    lab[:, :, 0] = (L / 100.0 * 255.0).astype(np.float32)
    lab[:, :, 1] = (a + 128).astype(np.float32)
    lab[:, :, 2] = (b + 128).astype(np.float32)
    
    # Convert back to RGB
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    rgb_uint8 = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
    
    return rgb_uint8.astype(np.float32) / 255.0


# =============================================================================
# VIBE REPLICATOR - Main Agentic Class
# =============================================================================

class VibeReplicator:
    """
    Agentic colorist that replicates a reference's vibe.
    
    Implements a 5-phase loop:
    1. analyze_vibe() - Understand the reference
    2. generate_instructions() - Create colorist instructions
    3. apply_instructions() - Grade the source image
    4. critique() - Self-assess the result
    5. refine_instructions() - Adjust if needed
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE Vibe Replicator"
        }
    
    def _call_llm(self, prompt: str, images: List[np.ndarray], max_tokens: int = 4000) -> Dict:
        """Make a call to the LLM with images."""
        image_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img)}"}}
            for img in images
        ]
        
        # Instruct Gemini 3 to be concise
        full_prompt = prompt + "\n\nIMPORTANT: Keep your reasoning very concise. Respond ONLY with valid JSON, no markdown formatting."
        
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
            "temperature": 0.3
        }
        
        try:
            print(f"[VIBE] Calling OpenRouter ({self.model})...")
            response = requests.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                print(f"[VIBE] ERROR: Unexpected response format: {result}")
                return {}
            
            content = result['choices'][0]['message'].get('content', '').strip()
            
            if not content:
                print(f"[VIBE] ERROR: Empty response content")
                return {}
            
            # Clean markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            # Try to find JSON in the response
            if '{' in content:
                start = content.index('{')
                # Find matching closing brace
                depth = 0
                end = start
                for i, c in enumerate(content[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                content = content[start:end]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"[VIBE] JSON parse error: {e}")
            print(f"[VIBE] Raw content: {content[:500]}...")
            return {}
        except Exception as e:
            print(f"[VIBE] LLM call failed: {e}")
            return {}
    
    # =========================================================================
    # PHASE 1: ANALYZE VIBE
    # =========================================================================
    
    def analyze_vibe(self, reference_img: np.ndarray) -> VibeAnalysis:
        """
        Phase 1: Analyze the reference image's vibe and color grading approach.
        
        Like a professional colorist first studying the reference before grading.
        """
        prompt = """You are a master colorist analyzing a reference image to understand its "vibe" and color grading.

Study this image and describe:

1. VIBE: What is the overall mood/feeling? (e.g., "Film noir tension", "Warm summer nostalgia", "Cold isolation")

2. MOOD KEYWORDS: 3-5 emotional keywords (e.g., ["tense", "mysterious", "dark"])

3. COLOR GRADING STYLE: Technical description of the grading (e.g., "Crushed blacks with teal shadows and orange highlights, desaturated midtones")

4. TECHNICAL OBSERVATIONS: What you notice about:
   - Shadow color cast (what color tint?)
   - Midtone treatment (neutral? shifted?)
   - Highlight color cast
   - Overall saturation level
   - Contrast style (flat? punchy? lifted blacks?)

Return JSON:
{
    "vibe_description": "<one sentence describing the vibe>",
    "mood_keywords": ["keyword1", "keyword2", "keyword3"],
    "color_grading_style": "<technical description of the grading approach>",
    "technical_observations": {
        "shadow_cast": "<color name and approximate Lab a,b shift>",
        "midtone_treatment": "<description>",
        "highlight_cast": "<color name and approximate Lab a,b shift>",
        "saturation_level": "<low/normal/high>",
        "contrast_style": "<description>"
    }
}"""

        print("\n[VIBE] Phase 1: Analyzing vibe...")
        result = self._call_llm(prompt, [reference_img])
        
        if not result:
            print("[VIBE] Vibe analysis failed, using defaults")
            return VibeAnalysis()
        
        analysis = VibeAnalysis(
            vibe_description=result.get("vibe_description", ""),
            mood_keywords=result.get("mood_keywords", []),
            color_grading_style=result.get("color_grading_style", ""),
            technical_observations=result.get("technical_observations", {})
        )
        
        print(f"[VIBE] Vibe: {analysis.vibe_description}")
        print(f"[VIBE] Style: {analysis.color_grading_style}")
        
        return analysis
    
    # =========================================================================
    # PHASE 2: GENERATE INSTRUCTIONS
    # =========================================================================
    
    def generate_instructions(
        self, 
        vibe_analysis: VibeAnalysis, 
        reference_img: np.ndarray
    ) -> GradingInstructions:
        """
        Phase 2: Generate explicit colorist instructions based on the vibe analysis.
        
        Converts the creative understanding into technical parameters.
        """
        prompt = f"""You are a master colorist. Based on this vibe analysis, generate PRECISE grading instructions:

VIBE ANALYSIS:
- Vibe: {vibe_analysis.vibe_description}
- Keywords: {', '.join(vibe_analysis.mood_keywords)}
- Style: {vibe_analysis.color_grading_style}
- Technical: {json.dumps(vibe_analysis.technical_observations)}

Looking at the reference image, generate EXACT numerical parameters to replicate this look:

Return JSON with these EXACT fields:
{{
    "exposure_adjustment": <float -1.0 to 1.0, EV adjustment>,
    "contrast_curve": "<s-curve|linear|lifted-blacks|crushed>",
    "contrast_strength": <float 0.8 to 1.5>,
    "shadow_a": <float -40 to 40, Lab a-axis shift for shadows>,
    "shadow_b": <float -40 to 40, Lab b-axis shift for shadows>,
    "midtone_a": <float -20 to 20, Lab a-axis shift for midtones>,
    "midtone_b": <float -20 to 20, Lab b-axis shift for midtones>,
    "highlight_a": <float -30 to 30, Lab a-axis shift for highlights>,
    "highlight_b": <float -30 to 30, Lab b-axis shift for highlights>,
    "saturation_mult": <float 0.5 to 1.5, overall saturation multiplier>,
    "special_notes": ["note1", "note2"],
    "description": "<one-line description of what these settings achieve>"
}}

LAB COLOR GUIDE:
- a axis: negative = green, positive = magenta/red
- b axis: negative = blue, positive = yellow

EXAMPLES:
- Matrix olive: shadow_a=-15, shadow_b=+10, midtone_a=-8, midtone_b=+5
- Orange/Teal: shadow_a=-10, shadow_b=-20, highlight_a=+10, highlight_b=+25
- Vintage warm: lifted-blacks, shadow_a=+5, shadow_b=+15, saturation_mult=0.8

Be BOLD with your values - subtle shifts don't create strong looks!"""

        print("[VIBE] Phase 2: Generating grading instructions...")
        result = self._call_llm(prompt, [reference_img])
        
        if not result:
            print("[VIBE] Instruction generation failed, using defaults")
            return GradingInstructions()
        
        instructions = GradingInstructions.from_dict(result)
        
        print(f"[VIBE] Instructions: {instructions.description}")
        print(f"[VIBE] Contrast: {instructions.contrast_curve} @ {instructions.contrast_strength}")
        print(f"[VIBE] Shadows: a={instructions.shadow_a}, b={instructions.shadow_b}")
        
        return instructions
    
    # =========================================================================
    # PHASE 3: APPLY INSTRUCTIONS
    # =========================================================================
    
    def apply_instructions(
        self, 
        source_img: np.ndarray, 
        instructions: GradingInstructions
    ) -> np.ndarray:
        """
        Phase 3: Apply the grading instructions to the source image.
        
        This is the mathematical application - no LLM needed here.
        """
        print("[VIBE] Phase 3: Applying grading instructions...")
        graded = apply_grading_instructions(source_img, instructions)
        print("[VIBE] Grade applied successfully")
        return graded
    
    # =========================================================================
    # PHASE 4: CRITIQUE
    # =========================================================================
    
    def critique(
        self, 
        reference_img: np.ndarray, 
        graded_img: np.ndarray,
        vibe_analysis: VibeAnalysis,
        instructions: GradingInstructions
    ) -> CritiqueResult:
        """
        Phase 4: Self-critique the graded result against the reference.
        
        The LLM compares its work to the target and suggests improvements.
        """
        prompt = f"""You are a master colorist performing a STRICT technical verification.

Step 1: REVERSE ENGINEER THE EDITS (Blind Analysis)
Look at the 'GRADED RESULT' (Image 2) vs the 'SOURCE' (not shown, but implied as the starting point).
- Guess EXACTLY what image editing choices have been made? (e.g., "Shadows were lifted and tinted teal", "Contrast was crunched").
- To which colors EXACTLY? (e.g., "The reds were desaturated", "The highlights were rolled off").

Step 2: DEDUCE PURPOSE & VIBE
- From these edits, what is the *intended* purpose? (e.g., "To create a cold, sterile medical look").
- Does this deduced intent match the ORIGINAL GOAL: "{vibe_analysis.vibe_description}"?

Step 3: STRICT COMPARISON (Output vs Reference)
Compare the 'GRADED RESULT' (Image 2) to the 'REFERENCE' (Image 1).
- Do the COLOR DISTRIBUTIONS match? (e.g., "Reference has 20% deep shadows, Result has 50% - too dark").
- Does the VIBE match? (Is the feeling successfully transported?)

Step 4: INSTRUCTIONS
If the match is not perfect (Score < 8), provide specific instructions on what to change.

CURRENT SETTINGS APPLIED:
{json.dumps(instructions.to_dict(), indent=2)}

Return JSON:
{{
    "analysis_of_edits": "<Your blind analysis of what edits observe in the result>",
    "vibe_match_score": <1-10, be strict! 10 = indistinguishable vibe>,
    "issues_found": ["Specific color mismatch 1", "Distribution mismatch 2"],
    "suggested_adjustments": {{
        "<parameter_name>": <new_value>,
        "<parameter_name>": <new_value>
    }},
    "satisfied": <true if score >= 8, otherwise false>,
    "feedback": "<Summary of why it passed or failed>"
}}

PARAMETER NAMES you can adjust:
- exposure_adjustment, contrast_curve, contrast_strength
- shadow_a, shadow_b, midtone_a, midtone_b, highlight_a, highlight_b
- saturation_mult"""

        print("[VIBE] Phase 4: Self-critiquing result (Strict Mode)...")
        result = self._call_llm(prompt, [reference_img, graded_img])
        
        if not result:
            print("[VIBE] Critique failed, assuming satisfied")
            return CritiqueResult(vibe_match_score=7, satisfied=True, feedback="Critique unavailable")
        
        critique = CritiqueResult(
            vibe_match_score=int(result.get("vibe_match_score", 5)),
            issues_found=result.get("issues_found", []),
            suggested_adjustments=result.get("suggested_adjustments", {}),
            satisfied=result.get("satisfied", False),
            feedback=result.get("feedback", ""),
            analysis_of_edits=result.get("analysis_of_edits", "")
        )
        
        print(f"[VIBE] Analysis: {critique.analysis_of_edits[:50]}...")
        print(f"[VIBE] Score: {critique.vibe_match_score}/10")
        print(f"[VIBE] Satisfied: {critique.satisfied}")
        
        return critique
    
    # =========================================================================
    # PHASE 5: REFINE INSTRUCTIONS
    # =========================================================================
    
    def refine_instructions(
        self, 
        instructions: GradingInstructions, 
        critique: CritiqueResult
    ) -> GradingInstructions:
        """
        Phase 5: Refine instructions based on the critique.
        
        Apply the suggested adjustments to create improved instructions.
        """
        print("[VIBE] Phase 5: Refining instructions based on critique...")
        
        # Create a copy and apply adjustments
        new_dict = instructions.to_dict()
        
        for param, value in critique.suggested_adjustments.items():
            if param in new_dict:
                new_dict[param] = value
                print(f"[VIBE] Adjusted {param}: {value}")
        
        # Update description with refinement note
        new_dict["description"] = f"Refined: {critique.feedback}"
        
        return GradingInstructions.from_dict(new_dict)
    
    # =========================================================================
    # FULL PIPELINE - Orchestrate all phases
    # =========================================================================
    
    def run_full_pipeline(
        self, 
        source_img: np.ndarray, 
        reference_img: np.ndarray,
        max_iterations: int = 3
    ) -> Dict:
        """
        Run the complete agentic loop.
        
        Returns:
            Dict containing:
            - vibe_analysis: VibeAnalysis object
            - final_instructions: GradingInstructions
            - graded_image: Final graded numpy array
            - critique_history: List of CritiqueResults
            - iterations: Number of iterations taken
        """
        print("\n" + "="*60)
        print("VIBE REPLICATOR: Starting Agentic Pipeline")
        print("="*60)
        
        # Phase 1: Analyze vibe
        vibe_analysis = self.analyze_vibe(reference_img)
        
        # Phase 2: Generate initial instructions
        instructions = self.generate_instructions(vibe_analysis, reference_img)
        
        critique_history = []
        graded_img = None
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Phase 3: Apply instructions
            graded_img = self.apply_instructions(source_img, instructions)
            
            # Phase 4: Critique
            critique = self.critique(reference_img, graded_img, vibe_analysis, instructions)
            critique_history.append(critique)
            
            if critique.satisfied:
                print(f"\n[VIBE] ✓ Satisfied after {iteration + 1} iteration(s)!")
                break
            
            if iteration < max_iterations - 1:
                # Phase 5: Refine for next iteration
                instructions = self.refine_instructions(instructions, critique)
        
        print("\n" + "="*60)
        print("VIBE REPLICATOR: Pipeline Complete")
        print(f"Final vibe match score: {critique_history[-1].vibe_match_score}/10")
        print("="*60 + "\n")
        
        return {
            "vibe_analysis": vibe_analysis,
            "final_instructions": instructions,
            "graded_image": graded_img,
            "critique_history": critique_history,
            "iterations": len(critique_history)
        }
