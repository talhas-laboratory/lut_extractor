import numpy as np
from skimage import color
from typing import Dict, List, Optional

class ColorCodeExtractor:
    """
    Extracts precise color codes from a reference image based on semantic composition.
    
    1. Receives semantic hints (roles + luma ranges) from SemanticGuide (LLM).
    2. Samples actual pixels from the reference image in those ranges.
    3. Computes robust statistics (median) to determine the true Lab target.
    """
    
    def extract_codebook(self, ref_img: np.ndarray, composition: Dict) -> Dict[str, Dict[str, float]]:
        """
        Generate a usable color codebook mapping roles to Lab values.
        
        Args:
            ref_img: Reference image (RGB float 0-1)
            composition: Output from SemanticGuide.analyze_composition()
            
        Returns:
            Dict[str, Dict[str, float]]: e.g. {'shadow': {'a': -5, 'b': -10}, ...}
        """
        # Convert reference to Lab
        ref_lab = color.rgb2lab(ref_img)
        L_channel = ref_lab[:, :, 0]
        
        codebook = {}
        
        elements = composition.get('elements', [])
        
        # If LLM didn't return useful elements, use defaults based on simple luma splitting
        if not elements:
            return self._extract_default_zones(ref_lab)
            
        for element in elements:
            role = element.get('role', 'unknown')
            luma_min = element.get('luma_min', 0)
            luma_max = element.get('luma_max', 100)
            
            # LLM suggested target values (fallback/guidance)
            llm_a = element.get('target_a', 0)
            llm_b = element.get('target_b', 0)
            
            # Create mask for this luma range
            # We also filter for 'chromatic' pixels if possible, to avoid sampling pure noise
            mask = (L_channel >= luma_min) & (L_channel <= luma_max)
            
            pixels = ref_lab[mask]
            
            if len(pixels) > 100:
                # Use actual image statistics from the masked region
                # Median is robust to outliers/noise
                measured_a = np.median(pixels[:, 1])
                measured_b = np.median(pixels[:, 2])
                
                # Blend LLM suggestion with measurement
                # We trust measurement more (80%) but keep LLM intent (20%)
                # This helps if the region is small or noisy
                final_a = 0.8 * measured_a + 0.2 * llm_a
                final_b = 0.8 * measured_b + 0.2 * llm_b
            else:
                # Fallback to LLM guess if no matching pixels found
                final_a = llm_a
                final_b = llm_b
            
            # Store in codebook
            # Use 'shadow', 'midtone', 'highlight' as primary keys if possible
            # Other keys like 'skin' or 'sky' stored as is
            codebook[role] = {'a': float(final_a), 'b': float(final_b)}
            
        # Ensure we have the basics even if LLM missed them or used weird names
        defaults = self._extract_default_zones(ref_lab)
        for key in ['shadow', 'midtone', 'highlight']:
            if key not in codebook:
                codebook[key] = defaults[key]
                
        # Also include global tint if present
        if 'global_tint' in composition:
            codebook['global'] = {
                'a': composition['global_tint'].get('a', 0),
                'b': composition['global_tint'].get('b', 0)
            }
            
        return codebook

    def _extract_default_zones(self, ref_lab: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Fallback: Extract simple shadow/midtone/highlight from luma ranges."""
        L = ref_lab[:, :, 0]
        
        # Shadow: 0-25
        mask_s = (L < 25)
        # Midtone: 25-75
        mask_m = (L >= 25) & (L <= 75)
        # Highlight: 75-100
        mask_h = (L > 75)
        
        def get_stat(mask):
            if np.sum(mask) < 10: return {'a': 0.0, 'b': 0.0}
            pixels = ref_lab[mask]
            return {
                'a': float(np.median(pixels[:, 1])), 
                'b': float(np.median(pixels[:, 2]))
            }
            
        return {
            'shadow': get_stat(mask_s),
            'midtone': get_stat(mask_m),
            'highlight': get_stat(mask_h)
        }
