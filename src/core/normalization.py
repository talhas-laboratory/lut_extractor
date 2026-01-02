"""
Normalization - Pre-grading exposure and white balance adjustments.

Applies exposure, temperature, and tint corrections to source images
before the main color grading pipeline runs. This ensures the source
is properly balanced relative to the reference's dynamic range.
"""

import numpy as np
from skimage import color
import cv2
from typing import Dict


def apply_normalization(src_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply exposure and white balance normalization to source image.
    
    This is the "prep" step before color grading - ensuring the source
    is balanced relative to the reference's base level.
    
    Args:
        src_img: RGB float32 [0,1]
        params: {
            "exposure": float (-1 to 1 stops),
            "temperature": float (-100 cool to 100 warm),
            "tint": float (-100 green to 100 magenta)
        }
    
    Returns:
        Normalized RGB image ready for grading pipeline.
    """
    exposure = params.get("exposure", 0.0)
    temperature = params.get("temperature", 0.0)
    tint = params.get("tint", 0.0)
    
    # Skip if no adjustments needed
    if abs(exposure) < 0.01 and abs(temperature) < 1 and abs(tint) < 1:
        return src_img.copy()
    
    result = src_img.copy()
    
    # 1. EXPOSURE ADJUSTMENT
    # Apply as a linear multiplier (stops to linear)
    if abs(exposure) > 0.01:
        multiplier = 2.0 ** exposure  # 1 stop = 2x brightness
        result = result * multiplier
        result = np.clip(result, 0, 1)
    
    # 2. WHITE BALANCE ADJUSTMENT (Temperature + Tint)
    # Work in Lab space for perceptually uniform adjustments
    if abs(temperature) > 1 or abs(tint) > 1:
        result_lab = color.rgb2lab(result)
        
        # Temperature: Affects b channel (blue-yellow axis)
        # Positive temperature = warmer = more yellow = positive b
        # Scale from -100/100 to approx -15/15 Lab units
        b_shift = temperature * 0.15
        
        # Tint: Affects a channel (green-magenta axis)  
        # Positive tint = more magenta = positive a
        # Scale from -100/100 to approx -10/10 Lab units
        a_shift = tint * 0.10
        
        result_lab[:, :, 1] += a_shift
        result_lab[:, :, 2] += b_shift
        
        # Clip Lab values
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)
        
        result = color.lab2rgb(result_lab)
        result = np.clip(result, 0, 1)
    
    return result.astype(np.float32)


def compute_exposure_difference(src_img: np.ndarray, ref_img: np.ndarray) -> float:
    """
    Compute the exposure difference between source and reference.
    
    This is a mathematical backup for when AI analysis is unavailable.
    Returns the difference in stops.
    """
    # Convert to Lab and compare L channel means
    src_lab = color.rgb2lab(src_img)
    ref_lab = color.rgb2lab(ref_img)
    
    src_l_mean = np.mean(src_lab[:, :, 0])
    ref_l_mean = np.mean(ref_lab[:, :, 0])
    
    # L is 0-100, convert difference to approximate stops
    # Rough heuristic: 15 L units ≈ 1 stop
    l_diff = ref_l_mean - src_l_mean
    stops = l_diff / 15.0
    
    # Clamp to reasonable range
    return np.clip(stops, -1.5, 1.5)


def compute_white_balance_difference(src_img: np.ndarray, ref_img: np.ndarray) -> Dict:
    """
    Compute white balance difference between source and reference.
    
    This is a mathematical backup for when AI analysis is unavailable.
    Returns temperature and tint values.
    """
    # Sample from midtones (L between 40-60) to avoid shadows/highlights bias
    src_lab = color.rgb2lab(src_img)
    ref_lab = color.rgb2lab(ref_img)
    
    # Mask for midtones
    src_midtone_mask = (src_lab[:, :, 0] > 40) & (src_lab[:, :, 0] < 60)
    ref_midtone_mask = (ref_lab[:, :, 0] > 40) & (ref_lab[:, :, 0] < 60)
    
    # Get median a,b values for midtones
    if np.sum(src_midtone_mask) > 100 and np.sum(ref_midtone_mask) > 100:
        src_a = np.median(src_lab[:, :, 1][src_midtone_mask])
        src_b = np.median(src_lab[:, :, 2][src_midtone_mask])
        ref_a = np.median(ref_lab[:, :, 1][ref_midtone_mask])
        ref_b = np.median(ref_lab[:, :, 2][ref_midtone_mask])
    else:
        # Fallback to full image
        src_a = np.median(src_lab[:, :, 1])
        src_b = np.median(src_lab[:, :, 2])
        ref_a = np.median(ref_lab[:, :, 1])
        ref_b = np.median(ref_lab[:, :, 2])
    
    # Compute difference and scale to -100/100 range
    # Note: We're computing what adjustment SOURCE needs, not the difference
    a_diff = ref_a - src_a  # Positive if ref is more magenta
    b_diff = ref_b - src_b  # Positive if ref is warmer/yellower
    
    # Scale from Lab units to -100/100 range
    temperature = b_diff / 0.15  # Inverse of the apply scaling
    tint = a_diff / 0.10
    
    return {
        "temperature": np.clip(temperature, -100, 100),
        "tint": np.clip(tint, -100, 100)
    }
