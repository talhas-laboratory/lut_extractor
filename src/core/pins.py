
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict

def compute_pins_with_labels(image: np.ndarray, n_clusters: int = 24) -> Tuple[np.ndarray, List[str]]:
    """
    Compute dominant color pins from an image using K-Means clustering,
    and append standard RGB boundary corners to ensure stability.
    Includes semantic labels for AI reasoning.

    Args:
        image (np.ndarray): Input image in shape (H, W, 3) or flattened (N, 3).
                            Should be float32 in range [0, 1].
        n_clusters (int): Number of K-Means centroids to extract. Default 24.

    Returns:
        Tuple[np.ndarray, List[str]]: 
            - Array of shape (n_clusters + 8, 3) containing the pins.
            - List of labels for each pin.
    """
    
    # Flatten image if it's (H, W, 3)
    if image.ndim == 3:
        pixels = image.reshape(-1, 3)
    else:
        pixels = image

    # Checks
    if pixels.dtype != np.float32 and pixels.dtype != np.float64:
        pixels = pixels.astype(np.float32)

    # 1. K-Means Clustering for Semantic Centroids
    if pixels.shape[0] > 10000:
        indices = np.random.choice(pixels.shape[0], 10000, replace=False)
        sample_pixels = pixels[indices]
    else:
        sample_pixels = pixels

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(sample_pixels)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    # Simple heuristic labels based on color
    labels = []
    for i, color in enumerate(centroids):
        r, g, b = color
        # Basic labeling logic
        if r > 0.7 and g > 0.7 and b > 0.7:
            labels.append(f"Highlight_{i}")
        elif r < 0.1 and g < 0.1 and b < 0.1:
            labels.append(f"Shadow_{i}")
        elif r > 0.6 and g > 0.4 and b > 0.3 and r > b: # Very basic skin heuristic
            labels.append(f"Skin_Tone_{i}")
        elif b > r and b > g:
            labels.append(f"Cool_Tone_{i}")
        elif g > r and g > b:
            labels.append(f"Green_Tone_{i}")
        elif r > g and r > b:
            labels.append(f"Warm_Tone_{i}")
        else:
            labels.append(f"Midtone_{i}")

    # 2. Add Identity/Boundary Pins (The 8 Corners of the RGB Cube)
    boundaries = np.array([
        [0.0, 0.0, 0.0], # Black
        [1.0, 1.0, 1.0], # White
        [1.0, 0.0, 0.0], # Red
        [0.0, 1.0, 0.0], # Green
        [0.0, 0.0, 1.0], # Blue
        [1.0, 1.0, 0.0], # Yellow
        [1.0, 0.0, 1.0], # Magenta
        [0.0, 1.0, 1.0]  # Cyan
    ], dtype=np.float32)
    
    boundary_labels = [
        "Pure_Black", "Pure_White", "Pure_Red", "Pure_Green", 
        "Pure_Blue", "Pure_Yellow", "Pure_Magenta", "Pure_Cyan"
    ]

    # Combine
    pins = np.vstack([centroids, boundaries])
    all_labels = labels + boundary_labels

    return pins, all_labels

def pins_to_json(pins: np.ndarray, labels: List[str]) -> List[Dict]:
    """Convert pins and labels to AI-ready JSON format."""
    return [
        {"id": i, "label": label, "r": float(p[0]), "g": float(p[1]), "b": float(p[2])}
        for i, (label, p) in enumerate(zip(labels, pins))
    ]

def json_to_pins(pin_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Convert AI-modified JSON back to numpy pins."""
    pins = []
    labels = []
    for item in pin_data:
        pins.append([item['r'], item['g'], item['b']])
        labels.append(item['label'])
    return np.array(pins, dtype=np.float32), labels


def apply_selective_shifts(
    source_pins: np.ndarray,
    target_pins: np.ndarray,
    operations: List[Dict]
) -> Dict[str, np.ndarray]:
    """
    Modify TPS pins based on selective correction operations from the AI Director.
    
    This is the bridge between Tool 4 (Selective Corrections) and the TPS engine.
    It modifies the target pins to implement region-specific adjustments.
    
    Supported operations:
    - "protect": Move target closer to source (preserve original color)
    - "shift_hue": Move target toward specified a,b values  
    - "desaturate": Reduce chroma magnitude in target
    - "saturate": Increase chroma magnitude in target
    
    Region detection is based on Lab ranges:
    - skin: 30 < L < 80, 5 < a < 30, 0 < b < 30
    - sky: 40 < L < 90, -30 < a < 0, -50 < b < 0 (blue region)
    - foliage: 20 < L < 70, -40 < a < -5, 10 < b < 60 (green region)
    - shadows: L < 25
    - highlights: L > 75
    
    Args:
        source_pins: Original source pins in Lab space [N, 3]
        target_pins: Current target pins in Lab space [N, 3]
        operations: List of operation dicts from SelectiveCorrectionParams
    
    Returns:
        {'source': source_pins (unchanged), 'target': modified_target_pins}
    """
    if not operations:
        return {'source': source_pins.copy(), 'target': target_pins.copy()}
    
    # Work with copies
    src = source_pins.copy()
    tgt = target_pins.copy()
    
    # Region detection masks based on Lab values
    # Pins are [L, a, b] in Lab space
    def get_region_mask(pins: np.ndarray, region: str) -> np.ndarray:
        """Get boolean mask for pins matching the specified region."""
        L = pins[:, 0]
        a = pins[:, 1]
        b = pins[:, 2]
        
        if region == "skin":
            # Skin tones: moderate L, positive a (red), positive b (yellow)
            return (L > 30) & (L < 80) & (a > 5) & (a < 30) & (b > 0) & (b < 30)
        
        elif region == "sky":
            # Blue sky: higher L, negative a (slightly green), negative b (blue)
            return (L > 40) & (L < 90) & (a > -30) & (a < 5) & (b < 0) & (b > -50)
        
        elif region == "foliage":
            # Greens: negative a (green), positive b (yellow-ish)
            return (L > 20) & (L < 70) & (a < -5) & (a > -40) & (b > 10) & (b < 60)
        
        elif region == "shadows":
            # Low luminance
            return L < 25
        
        elif region == "highlights":
            # High luminance
            return L > 75
        
        else:
            # Unknown region - return empty mask
            return np.zeros(len(pins), dtype=bool)
    
    # Apply each operation
    for op in operations:
        if isinstance(op, dict):
            region = op.get("region", "")
            action = op.get("action", "")
            strength = float(op.get("strength", 1.0))
            target_a = float(op.get("target_a", 0))
            target_b = float(op.get("target_b", 0))
        else:
            # Assume it's a SelectiveOperation dataclass
            region = op.region
            action = op.action
            strength = op.strength
            target_a = op.target_a
            target_b = op.target_b
        
        # Get mask for affected pins (based on SOURCE pin locations)
        mask = get_region_mask(src, region)
        
        if not np.any(mask):
            continue
        
        if action == "protect":
            # Move target closer to source (reduce the color shift)
            # At strength=1.0, target becomes source (full protection)
            # At strength=0.5, target moves halfway toward source
            tgt[mask] = tgt[mask] + strength * (src[mask] - tgt[mask])
        
        elif action == "shift_hue":
            # Move target's a,b toward specified values
            # Only modify chroma (a,b), preserve L
            current_a = tgt[mask, 1]
            current_b = tgt[mask, 2]
            
            tgt[mask, 1] = current_a + strength * (target_a - current_a)
            tgt[mask, 2] = current_b + strength * (target_b - current_b)
        
        elif action == "desaturate":
            # Reduce chroma magnitude (move a,b toward 0)
            reduction = 1.0 - strength  # strength=1 means fully desaturate
            tgt[mask, 1] *= reduction
            tgt[mask, 2] *= reduction
        
        elif action == "saturate":
            # Increase chroma magnitude (amplify a,b)
            amplification = 1.0 + strength  # strength=0.5 means 50% boost
            tgt[mask, 1] *= amplification
            tgt[mask, 2] *= amplification
            # Clip to valid Lab range
            tgt[mask, 1] = np.clip(tgt[mask, 1], -128, 127)
            tgt[mask, 2] = np.clip(tgt[mask, 2], -128, 127)
    
    return {'source': src, 'target': tgt}


from skimage import color

def create_palette_pins_from_casts(
    global_casts: Dict,
    n_l_levels: int = 9  # Increased resolution for smoother curves
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create TPS pins from AI-extracted palette identity using ZONE SEPARATION.
    
    Instead of linear blending (which creates a muddy wash), this uses 
    sigmoid/gaussian weighting to isolate Shadow, Midtone, and Highlight zones.
    
    Args:
        global_casts: {
            "shadow_cast": {"a": float, "b": float},
            "highlight_cast": {"a": float, "b": float},
            "neutral_shift": {"a": float, "b": float},
            "mid_cast": {"a": float, "b": float} (Optional explicit midtone)
        }
        n_l_levels: Number of luminance levels to sample
    
    Returns:
        (source_pins, target_pins) in Lab space
    """
    shadow_cast = global_casts.get("shadow_cast", {"a": 0, "b": 0})
    highlight_cast = global_casts.get("highlight_cast", {"a": 0, "b": 0})
    # Use neutral_shift as the base midtone cast if mid_cast isn't explicit
    mid_cast = global_casts.get("mid_cast", global_casts.get("neutral_shift", {"a": 0, "b": 0}))
    
    source_pins = []
    target_pins = []
    
    # Create pins at various L levels (0 to 100)
    l_levels = np.linspace(0, 100, n_l_levels)
    
    def sigmoid(x, center, steepness):
        """Soft switching function"""
        return 1 / (1 + np.exp(-steepness * (x - center)))

    def gaussian(x, center, width):
        """Bell curve for isolating specific ranges (like midtones)"""
        return np.exp(-0.5 * ((x - center) / width) ** 2)
    
    for L in l_levels:
        # Source pin is neutral gray at this L
        source_pins.append([L, 0, 0])
        
        # ZONE WEIGHT CALCULATION
        # 1. Shadow influence: Strong below L=30, fades out by L=50
        w_shadow = 1 - sigmoid(L, center=30, steepness=0.15)
        
        # 2. Highlight influence: Strong above L=70, fades in from L=50
        w_highlight = sigmoid(L, center=70, steepness=0.15)
        
        # 3. Midtone influence: Captures whatever is left in the middle
        # Normalize weights so they sum to 1 (approximately)
        # We model midtone as the 'bridge' between shadow and highlight
        w_total = w_shadow + w_highlight
        w_mid = max(0, 1 - w_total)
        
        # Re-normalize to ensure exact sum to 1.0
        norm = w_shadow + w_highlight + w_mid
        w_shadow /= norm
        w_highlight /= norm
        w_mid /= norm
        
        # Calculate resulting Tint
        target_a = (shadow_cast["a"] * w_shadow + 
                   mid_cast["a"] * w_mid + 
                   highlight_cast["a"] * w_highlight)
                   
        target_b = (shadow_cast["b"] * w_shadow + 
                   mid_cast["b"] * w_mid + 
                   highlight_cast["b"] * w_highlight)
        
        target_pins.append([L, target_a, target_b])
    
    # NEW: Add RGB Chromatic Corners to prevent Singular Matrix in TPS
    # (TPS fails if all points are collinear on the neutral axis)
    rgb_corners = np.array([
        [1, 0, 0], # Red
        [0, 1, 0], # Green
        [0, 0, 1], # Blue
        [1, 1, 0], # Yellow
        [1, 0, 1], # Magenta
        [0, 1, 1]  # Cyan
    ], dtype=np.float32)
    
    # Convert RGB corners to Lab
    # shape (N, 3) -> (1, N, 3) for rgb2lab -> (N, 3)
    lab_corners = color.rgb2lab(rgb_corners.reshape(1, -1, 3)).reshape(-1, 3)
    
    for lab_corner in lab_corners:
        source_pins.append(lab_corner)
        target_pins.append(lab_corner) # Boundaries are fixed (identity)
    
    return np.array(source_pins, dtype=np.float32), np.array(target_pins, dtype=np.float32)

