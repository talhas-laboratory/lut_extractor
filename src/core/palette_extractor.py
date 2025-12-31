"""
PaletteExtractor - Extract color transformation from reference images.

The key insight: Instead of assuming source and reference have similar content,
we extract the reference's COLOR PALETTE and TRANSFORMATION STYLE directly:
- How neutrals are shifted (the "tint")
- How different luminance zones are colored
- Hue rotation applied to the entire image
"""

import numpy as np
from sklearn.cluster import KMeans
from skimage import color
import cv2
from typing import Dict, List, Tuple, Optional


class PaletteExtractor:
    """
    Extracts the colorist's style from a reference image.
    Works by analyzing how colors relate to each other,
    not by assuming spatial correspondence.
    """
    
    def __init__(self, n_palette_colors: int = 48):
        self.n_palette_colors = n_palette_colors
        self.palette = None
        self.hue_rotation = 0.0
        self.zone_tints = None
        
    def analyze_reference(self, ref_img: np.ndarray) -> Dict:
        """
        Full analysis of the reference image's color style.
        
        Returns:
            Dict with all extracted color information
        """
        # Convert to Lab and HLS for analysis
        ref_lab = color.rgb2lab(ref_img)
        ref_hls = cv2.cvtColor((ref_img * 255).astype(np.uint8), cv2.COLOR_RGB2HLS)
        
        results = {
            'palette': self.extract_palette(ref_lab),
            'hue_rotation': self.compute_hue_rotation(ref_lab),
            'zone_tints': self.compute_zone_tints(ref_lab),
            'neutral_shift': self.compute_neutral_shift(ref_lab),
            'color_stats': self.compute_color_stats(ref_lab)
        }
        
        self.palette = results['palette']
        self.hue_rotation = results['hue_rotation']
        self.zone_tints = results['zone_tints']
        
        return results
    
    def extract_palette(self, ref_lab: np.ndarray) -> List[Dict]:
        """
        Extract dominant colors with their luminance zones.
        
        Returns:
            List of {L_zone, a, b, weight, rgb_approx}
        """
        h, w, _ = ref_lab.shape
        flat = ref_lab.reshape(-1, 3)
        
        # Subsample for clustering
        n_samples = min(50000, len(flat))
        indices = np.random.choice(len(flat), n_samples, replace=False)
        samples = flat[indices]
        
        # Cluster to find palette
        kmeans = KMeans(n_clusters=self.n_palette_colors, n_init='auto', random_state=42)
        kmeans.fit(samples)
        
        # Get cluster info
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        palette = []
        for i, center in enumerate(centers):
            L, a, b = center
            count = np.sum(labels == i)
            weight = count / len(labels)
            
            # Determine luminance zone
            if L < 30:
                zone = 'shadow'
            elif L < 60:
                zone = 'midtone'
            else:
                zone = 'highlight'
            
            # Convert to RGB for visualization
            lab_color = np.array([[[L, a, b]]], dtype=np.float64)
            rgb_approx = color.lab2rgb(lab_color)[0, 0]
            
            palette.append({
                'L': float(L),
                'a': float(a),
                'b': float(b),
                'zone': zone,
                'weight': float(weight),
                'rgb': rgb_approx.tolist()
            })
        
        # Sort by weight
        palette.sort(key=lambda x: -x['weight'])
        return palette
    
    def compute_hue_rotation(self, ref_lab: np.ndarray) -> float:
        """
        Detect how much neutral grays are shifted.
        
        The Matrix shifts neutral grays toward cyan-green.
        We detect this by finding low-saturation pixels and 
        measuring their a,b offset.
        """
        flat = ref_lab.reshape(-1, 3)
        
        # Find "near neutral" pixels (low chroma in Lab)
        # Chroma = sqrt(a² + b²)
        a, b = flat[:, 1], flat[:, 2]
        chroma = np.sqrt(a**2 + b**2)
        
        # Midtone luminance (where tinting is most visible)
        L = flat[:, 0]
        midtone_mask = (L > 25) & (L < 75)
        
        # Low saturation (would be neutral in ungraded footage)
        # In graded footage, "neutral" may have chroma up to ~20
        low_sat_mask = chroma < 25
        
        # Combined mask
        neutral_candidates = midtone_mask & low_sat_mask
        
        if np.sum(neutral_candidates) < 100:
            # Not enough neutral pixels, return 0
            return 0.0
        
        # Average a,b of "neutral" pixels
        avg_a = np.mean(a[neutral_candidates])
        avg_b = np.mean(b[neutral_candidates])
        
        # Convert to hue angle (degrees)
        # In Lab: hue = atan2(b, a) * 180 / pi
        hue_angle = np.degrees(np.arctan2(avg_b, avg_a))
        
        # The "shift" from neutral (which would be 0,0)
        # Negative a = green, Negative b = blue
        # Matrix look: a ≈ -8, b ≈ -15 (cyan-green)
        
        print(f"  [Palette] Neutral shift detected: a={avg_a:.2f}, b={avg_b:.2f}, hue={hue_angle:.1f}°")
        
        return hue_angle
    
    def compute_neutral_shift(self, ref_lab: np.ndarray) -> Dict[str, float]:
        """
        Get the exact a,b shift applied to neutrals.
        This is applied globally to bring grays toward the reference tint.
        """
        flat = ref_lab.reshape(-1, 3)
        
        L = flat[:, 0]
        a, b = flat[:, 1], flat[:, 2]
        chroma = np.sqrt(a**2 + b**2)
        
        # Focus on midtones with low chroma
        midtone_mask = (L > 30) & (L < 70)
        low_chroma_mask = chroma < 20
        mask = midtone_mask & low_chroma_mask
        
        if np.sum(mask) < 50:
            return {'a': 0.0, 'b': 0.0}
        
        return {
            'a': float(np.mean(a[mask])),
            'b': float(np.mean(b[mask]))
        }
    
    def compute_zone_tints(self, ref_lab: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute the average tint (a,b) for each luminance zone.
        This captures how shadows, midtones, and highlights are colored differently.
        """
        flat = ref_lab.reshape(-1, 3)
        L = flat[:, 0]
        a, b = flat[:, 1], flat[:, 2]
        
        zones = {
            'shadow': (0, 30),
            'lowmid': (30, 45),
            'midtone': (45, 60),
            'highmid': (60, 75),
            'highlight': (75, 100)
        }
        
        zone_tints = {}
        for zone_name, (L_min, L_max) in zones.items():
            mask = (L >= L_min) & (L < L_max)
            if np.sum(mask) > 100:
                zone_tints[zone_name] = {
                    'a': float(np.mean(a[mask])),
                    'b': float(np.mean(b[mask])),
                    'L_avg': float(np.mean(L[mask])),
                    'chroma_avg': float(np.mean(np.sqrt(a[mask]**2 + b[mask]**2)))
                }
            else:
                zone_tints[zone_name] = {'a': 0, 'b': 0, 'L_avg': (L_min + L_max) / 2, 'chroma_avg': 0}
        
        return zone_tints
    
    def compute_color_stats(self, ref_lab: np.ndarray) -> Dict:
        """Overall color statistics."""
        flat = ref_lab.reshape(-1, 3)
        
        return {
            'L_mean': float(np.mean(flat[:, 0])),
            'L_std': float(np.std(flat[:, 0])),
            'a_mean': float(np.mean(flat[:, 1])),
            'a_std': float(np.std(flat[:, 1])),
            'b_mean': float(np.mean(flat[:, 2])),
            'b_std': float(np.std(flat[:, 2])),
            'chroma_mean': float(np.mean(np.sqrt(flat[:, 1]**2 + flat[:, 2]**2)))
        }


def compute_tps_from_palette(
    palette_analysis: Dict,
    n_pins: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build TPS control points from palette analysis.
    
    Instead of spatial correspondence, we create mappings:
    - Neutral gray at each L level → reference zone tint at that L
    - Key palette colors are preserved as anchors
    
    Returns:
        (source_pins, target_pins) both in Lab space
    """
    zone_tints = palette_analysis['zone_tints']
    neutral_shift = palette_analysis['neutral_shift']
    
    source_pins = []
    target_pins = []
    
    # Create neutral → tinted mappings across L range
    for L in np.linspace(5, 95, 19):  # Every 5 units of luminance
        # Source: neutral gray at this L
        src = [L, 0, 0]
        
        # Target: apply zone-appropriate tint
        if L < 30:
            zone = 'shadow'
        elif L < 45:
            zone = 'lowmid'
        elif L < 60:
            zone = 'midtone'
        elif L < 75:
            zone = 'highmid'
        else:
            zone = 'highlight'
        
        tint = zone_tints.get(zone, {'a': 0, 'b': 0})
        tgt = [L, tint['a'], tint['b']]
        
        source_pins.append(src)
        target_pins.append(tgt)
    
    # Add some chroma points to preserve saturation relationships
    # For each zone, add a saturated point that maps to slightly more/less saturated
    chroma_offsets = [
        # (L, src_a, src_b, tgt_a_mult, tgt_b_mult)
        (40, 20, 0, 1.0, 1.0),   # Red-ish midtone
        (40, -20, 0, 1.0, 1.0),  # Green midtone
        (40, 0, 20, 1.0, 1.0),   # Yellow midtone
        (40, 0, -20, 1.0, 1.0),  # Blue midtone
        (60, 15, 15, 1.0, 1.0),  # Warm highlight
        (25, -10, -10, 1.0, 1.0), # Cool shadow
    ]
    
    for L, src_a, src_b, a_mult, b_mult in chroma_offsets:
        # Determine zone tint
        if L < 30:
            zone = 'shadow'
        elif L < 60:
            zone = 'midtone'
        else:
            zone = 'highlight'
        
        zone_tint = zone_tints.get(zone, {'a': 0, 'b': 0})
        
        src = [L, src_a, src_b]
        # Target adds zone tint to the colored point
        tgt = [L, src_a * a_mult + zone_tint['a'], src_b * b_mult + zone_tint['b']]
        
        source_pins.append(src)
        target_pins.append(tgt)
    
    # Add boundary constraints
    source_pins.append([0, 0, 0])   # Pure black
    source_pins.append([100, 0, 0]) # Pure white
    target_pins.append([0, 0, 0])   # Black stays black
    target_pins.append([100, 0, 0]) # White stays white
    
    return np.array(source_pins, dtype=np.float32), np.array(target_pins, dtype=np.float32)


def apply_zone_tints(
    img_lab: np.ndarray,
    zone_tints: Dict[str, Dict[str, float]],
    strength: float = 1.0
) -> np.ndarray:
    """
    Apply zone-specific tinting to a Lab image.
    
    This is a fast alternative to TPS for applying the reference look.
    """
    result = img_lab.copy()
    L = result[:, :, 0]
    
    # Create smooth weight masks for each zone
    # Using smooth transitions avoids banding
    
    def zone_weight(L_val, center, width):
        """Gaussian-like weight centered at L value."""
        return np.exp(-0.5 * ((L_val - center) / width) ** 2)
    
    # Apply each zone's tint with weighted blending
    zones = {
        'shadow': (15, 15),      # center=15, width=15
        'lowmid': (37.5, 10),
        'midtone': (52.5, 10),
        'highmid': (67.5, 10),
        'highlight': (87.5, 15)
    }
    
    a_shift = np.zeros_like(L)
    b_shift = np.zeros_like(L)
    total_weight = np.zeros_like(L)
    
    for zone_name, (center, width) in zones.items():
        tint = zone_tints.get(zone_name, {'a': 0, 'b': 0})
        weight = zone_weight(L, center, width)
        
        a_shift += weight * tint['a']
        b_shift += weight * tint['b']
        total_weight += weight
    
    # Normalize and apply
    total_weight = np.maximum(total_weight, 1e-6)
    a_shift = (a_shift / total_weight) * strength
    b_shift = (b_shift / total_weight) * strength
    
    result[:, :, 1] = np.clip(result[:, :, 1] + a_shift, -128, 127)
    result[:, :, 2] = np.clip(result[:, :, 2] + b_shift, -128, 127)
    
    return result
