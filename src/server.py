
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import uuid
import numpy as np
import cv2
from scipy.interpolate import interp1d
from skimage import color
from typing import List, Tuple, Dict


# Import project modules
from src.io.loader import load_image, save_image
from src.ai.color_scientist import ColorScientist, apply_ai_adjustments
from src.ai.agentic_critic import AgenticCritic
from src.ai.semantic_guide import SemanticGuide
from src.core.tps import ThinPlateSpline
from src.core.pins import compute_pins_with_labels, pins_to_json, json_to_pins
from src.core.palette_extractor import PaletteExtractor, compute_tps_from_palette, apply_zone_tints
from src.core.color_codebook import ColorCodeExtractor
from src.io.baker import generate_identity_lattice, export_to_cube

app = FastAPI(title="HCGE Backend API - Hybrid Luma-Chroma Engine")

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class PipelineState:
    """In-memory session storage. Use Redis/DB in production."""
    sessions = {}


def compute_zoned_luma_mapper(
    src_img: np.ndarray, 
    ref_img: np.ndarray, 
    strength: float = 0.6,
    shadow_treatment: str = 'preserve',
    highlight_treatment: str = 'preserve'
) -> interp1d:
    """
    ENHANCED: Zone-based luma transfer function.
    
    Matches luminance SEPARATELY for shadows, midtones, and highlights.
    This preserves the reference's tonal zones without flattening.
    
    For Matrix: Crush blacks while preserving midtone brightness.
    """
    # Convert to Lab
    src_lab = color.rgb2lab(src_img)
    ref_lab = color.rgb2lab(ref_img)
    
    src_l = src_lab[:, :, 0].flatten()
    ref_l = ref_lab[:, :, 0].flatten()
    
    # Compute per-zone statistics
    def zone_stats(l_flat, low, high):
        mask = (l_flat >= low) & (l_flat < high)
        if np.sum(mask) > 100:
            return np.mean(l_flat[mask]), np.std(l_flat[mask])
        return (low + high) / 2, 15  # Default
    
    zones = [
        (0, 25, 'shadow'),
        (25, 45, 'lowmid'),
        (45, 65, 'midtone'),
        (65, 85, 'highmid'),
        (85, 100, 'highlight')
    ]
    
    src_stats = {}
    ref_stats = {}
    for low, high, name in zones:
        src_stats[name] = zone_stats(src_l, low, high)
        ref_stats[name] = zone_stats(ref_l, low, high)
    
    print(f"  Zone-based Luma Analysis:")
    for name in ['shadow', 'midtone', 'highlight']:
        sm, ss = src_stats[name]
        rm, rs = ref_stats[name]
        print(f"    {name}: src={sm:.1f}±{ss:.1f}, ref={rm:.1f}±{rs:.1f}")
    
    # Build piecewise transfer curve
    l_values = np.linspace(0, 100, 256)
    final_curve = np.zeros_like(l_values)
    
    for low, high, name in zones:
        mask = (l_values >= low) & (l_values < high)
        src_mean, src_std = src_stats[name]
        ref_mean, ref_std = ref_stats[name]
        
        # Zone-specific strength modifiers
        zone_strength = strength
        if name == 'shadow' and shadow_treatment == 'crush':
            zone_strength = min(1.0, strength + 0.3)  # Crush blacks harder
        elif name == 'highlight' and highlight_treatment == 'roll':
            zone_strength = min(1.0, strength + 0.2)  # Roll off highlights
        
        # Apply normalized transfer for this zone
        zone_vals = l_values[mask]
        if src_std > 0:
            normalized = (zone_vals - src_mean) / src_std
        else:
            normalized = zone_vals - src_mean
        
        blended_mean = src_mean + (ref_mean - src_mean) * zone_strength
        blended_std = src_std + (ref_std - src_std) * zone_strength * 0.7
        
        transferred = normalized * blended_std + blended_mean
        
        # Blend with identity
        identity = zone_vals
        final_curve[mask] = zone_strength * transferred + (1 - zone_strength) * identity
    
    # Smooth zone transitions using small gaussian blur on the curve
    from scipy.ndimage import gaussian_filter1d
    final_curve = gaussian_filter1d(final_curve, sigma=3)
    
    # Clip to valid range
    final_curve = np.clip(final_curve, 0, 100)
    
    # Create interpolation function
    luma_mapper = interp1d(l_values, final_curve, kind='linear', 
                           bounds_error=False, fill_value=(final_curve[0], final_curve[-1]))
    
    return luma_mapper


# Keep the old function for backward compatibility
def compute_luma_mapper(src_img: np.ndarray, ref_img: np.ndarray, strength: float = 0.6) -> interp1d:
    """Legacy function - calls the new zoned version with defaults."""
    return compute_zoned_luma_mapper(src_img, ref_img, strength)



def compute_chroma_tps_from_pins(source_pins: np.ndarray, target_pins: np.ndarray) -> ThinPlateSpline:
    """Helper to fit TPS from pre-computed pins."""
    tps = ThinPlateSpline(smoothing=0.01)
    tps.fit(source_points=source_pins, target_points=target_pins)
    return tps

def compute_chroma_tps(source_img: np.ndarray, ref_img: np.ndarray, n_clusters: int = 32) -> Tuple[ThinPlateSpline, np.ndarray, np.ndarray, List[str]]:
    """
    Compute TPS warp for colors from Source → Reference in Lab space.
    
    CRITICAL: Uses SPATIAL CORRESPONDENCE - we compare the same pixel positions
    between source and reference, then cluster to find how colors transformed.
    This prevents mismatches like "source sky → reference mountain".
    """
    from sklearn.cluster import KMeans
    
    # Resize source to match reference dimensions for spatial correspondence
    if source_img.shape[:2] != ref_img.shape[:2]:
        source_resized = cv2.resize(source_img, (ref_img.shape[1], ref_img.shape[0]))
    else:
        source_resized = source_img
    
    # Convert BOTH to Lab at the same resolution
    src_lab = color.rgb2lab(source_resized)
    ref_lab = color.rgb2lab(ref_img)
    
    h, w, _ = src_lab.shape
    src_flat = src_lab.reshape(-1, 3)
    ref_flat = ref_lab.reshape(-1, 3)
    
    # Subsample for clustering (same indices = spatial correspondence!)
    n_pixels = len(src_flat)
    sample_size = min(50000, n_pixels)
    indices = np.random.choice(n_pixels, sample_size, replace=False)
    
    # Cluster on SOURCE colors to find palette centers
    print(f"  Clustering {n_clusters} color groups in Lab space...")
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(src_flat[indices])
    
    labels = kmeans.predict(src_flat[indices])
    
    # Build pin correspondences using SPATIAL matching
    # For each cluster in source, find what those SAME PIXELS became in reference
    source_pins = []
    target_pins = []
    pin_labels = []
    
    for i in range(n_clusters):
        mask = (labels == i)
        if np.sum(mask) < 20:
            continue
        
        # SAME pixels in both images (this is the key!)
        src_colors = src_flat[indices][mask]
        ref_colors = ref_flat[indices][mask]  # Same pixel positions!
        
        # Average Lab for this cluster
        avg_src = np.mean(src_colors, axis=0)
        avg_ref = np.mean(ref_colors, axis=0)
        
        source_pins.append(avg_src)
        target_pins.append(avg_ref)
        
        # Generate semantic label
        L, a, b = avg_src
        if L > 70:
            label = f"Highlight_{i}"
        elif L < 30:
            label = f"Shadow_{i}"
        elif a > 10:
            label = f"Warm_{i}"
        elif a < -10:
            label = f"Cool_{i}"
        else:
            label = f"Midtone_{i}"
        pin_labels.append(label)
    
    source_pins = np.array(source_pins, dtype=np.float32)
    target_pins = np.array(target_pins, dtype=np.float32)
    
    print(f"  Found {len(source_pins)} Lab color correspondences (spatial matching)")
    
    # Add boundary constraints (black/white stay fixed)
    lab_bounds_src = np.array([
        [0, 0, 0],    # Pure black
        [100, 0, 0],  # Pure white
    ], dtype=np.float32)
    lab_bounds_tgt = lab_bounds_src.copy()
    
    source_pins = np.vstack([source_pins, lab_bounds_src])
    target_pins = np.vstack([target_pins, lab_bounds_tgt])
    pin_labels.extend(["Pure_Black", "Pure_White"])
    
    # Fit 3D TPS in Lab space
    print(f"  Fitting 3D TPS for Lab warp (smoothing=0.01)...")
    tps = ThinPlateSpline(smoothing=0.01)
    tps.fit(source_pins, target_pins)
    
    return tps, source_pins, target_pins, pin_labels


def compute_palette_based_tps(
    ref_img: np.ndarray, 
    palette_analysis: Dict = None,
    semantic_hints: Dict = None
) -> Tuple[ThinPlateSpline, np.ndarray, np.ndarray, List[str], Dict]:
    """
    NEW: Compute TPS from REFERENCE-ONLY palette analysis.
    
    This doesn't assume source and reference have similar content.
    Instead, we extract HOW the colorist graded the reference:
    - What do neutrals become? (the tint)
    - How are shadows colored?
    - How are highlights colored?
    
    Then we apply that same transformation to ANY source image.
    """
    from src.core.palette_extractor import PaletteExtractor, compute_tps_from_palette
    
    # Extract palette from reference if not provided
    if palette_analysis is None:
        extractor = PaletteExtractor(n_palette_colors=48)
        palette_analysis = extractor.analyze_reference(ref_img)
    
    # Merge with semantic hints if available
    if semantic_hints is not None:
        # Use LLM hints to validate/adjust the math
        llm_tints = semantic_hints.get('zone_tints', {})
        math_tints = palette_analysis.get('zone_tints', {})
        
        # If LLM detected a specific look (e.g., "Matrix green"), trust its direction
        dominant_a = semantic_hints.get('dominant_hue_lab_a', 0)
        dominant_b = semantic_hints.get('dominant_hue_lab_b', 0)
        
        # If LLM found a strong tint, blend it with math
        if abs(dominant_a) > 5 or abs(dominant_b) > 5:
            for zone in ['shadow', 'midtone', 'highlight']:
                if zone in math_tints:
                    # 60% math, 40% LLM (LLM guides but math measures)
                    llm_zone = llm_tints.get(zone, {'a': dominant_a, 'b': dominant_b})
                    math_tints[zone]['a'] = 0.6 * math_tints[zone]['a'] + 0.4 * llm_zone.get('a', dominant_a)
                    math_tints[zone]['b'] = 0.6 * math_tints[zone]['b'] + 0.4 * llm_zone.get('b', dominant_b)
            
            palette_analysis['zone_tints'] = math_tints
    
    # Build TPS pins from palette analysis
    source_pins, target_pins = compute_tps_from_palette(palette_analysis)
    
    # Generate labels
    pin_labels = []
    for i, (src, tgt) in enumerate(zip(source_pins, target_pins)):
        L = src[0]
        if L < 20:
            pin_labels.append(f"Shadow_L{int(L)}")
        elif L > 80:
            pin_labels.append(f"Highlight_L{int(L)}")
        else:
            pin_labels.append(f"Mid_L{int(L)}")
    
    print(f"  [Palette TPS] Generated {len(source_pins)} pins from reference palette")
    print(f"  [Palette TPS] Neutral shift: a={palette_analysis.get('neutral_shift', {}).get('a', 0):.1f}, b={palette_analysis.get('neutral_shift', {}).get('b', 0):.1f}")
    
    # Fit TPS
    tps = ThinPlateSpline(smoothing=0.01)
    tps.fit(source_pins, target_pins)
    
    return tps, source_pins, target_pins, pin_labels, palette_analysis


def apply_hybrid_grade(src_img: np.ndarray, luma_mapper: interp1d, chroma_tps: ThinPlateSpline, 
                        saturation_boost: float = 1.2) -> np.ndarray:
    """
    Apply the hybrid grade to an image:
    1. Luma: Apply histogram-matched L from User→Reference
    2. Chroma: Apply TPS-warped a,b from Proxy→Reference mapping
    3. Saturation boost: Amplify chroma differences for richer colors
    """
    # Convert to Lab
    src_lab = color.rgb2lab(src_img)
    h, w, _ = src_lab.shape
    
    # 1. LUMA ANCHOR: Map L channel using histogram matching
    src_l = src_lab[:, :, 0]
    matched_l = luma_mapper(src_l)
    matched_l = np.clip(matched_l, 0, 100)
    
    # 2. CHROMA WARP: Apply TPS to full Lab, but only use a,b from result
    src_lab_flat = src_lab.reshape(-1, 3)
    warped_lab_flat = chroma_tps.transform(src_lab_flat)
    warped_lab_flat = np.clip(warped_lab_flat, [0, -128, -128], [100, 127, 127])
    warped_lab = warped_lab_flat.reshape(h, w, 3)
    
    # 3. SATURATION BOOST: Amplify the a,b channels for richer colors
    # This helps recover the vibrant blues/teals from the reference
    warped_a = warped_lab[:, :, 1] * saturation_boost
    warped_b = warped_lab[:, :, 2] * saturation_boost
    warped_a = np.clip(warped_a, -128, 127)
    warped_b = np.clip(warped_b, -128, 127)
    
    # 4. COMBINE: Matched L (from histogram) + Boosted warped a,b (from TPS)
    graded_lab = np.zeros_like(src_lab)
    graded_lab[:, :, 0] = matched_l     # L from histogram matching
    graded_lab[:, :, 1] = warped_a      # Boosted a from TPS
    graded_lab[:, :, 2] = warped_b      # Boosted b from TPS
    
    # Convert back to RGB
    graded_rgb = color.lab2rgb(graded_lab)
    graded_rgb = np.clip(graded_rgb, 0, 1).astype(np.float32)
    
    return graded_rgb


def apply_direct_color_grade(
    src_img: np.ndarray, 
    luma_mapper: interp1d, 
    color_codebook: Dict[str, Dict[str, float]],
    saturation_mult: float = 0.85
) -> np.ndarray:
    """
    COMPOSITIONAL APPROACH: Apply colors directly from a codebook.
    
    Instead of TPS warping, we apply zone-specific Lab tints directly.
    This is simpler, more predictable, and captures the exact reference colors.
    
    Args:
        src_img: Source image (RGB float32 0-1)
        luma_mapper: Luma transfer function
        color_codebook: Dict mapping zones to Lab a,b values
            e.g., {"shadow": {"a": -10, "b": -15}, "midtone": {"a": -8, "b": -12}, ...}
        saturation_mult: Reduce saturation to match desaturated looks
    """
    # Convert to Lab
    src_lab = color.rgb2lab(src_img)
    h, w, _ = src_lab.shape
    
    # 1. LUMA: Apply histogram-matched L
    src_l = src_lab[:, :, 0]
    matched_l = luma_mapper(src_l)
    matched_l = np.clip(matched_l, 0, 100)
    
    # 2. DESATURATE slightly (Matrix look is desaturated)
    src_a = src_lab[:, :, 1] * saturation_mult
    src_b = src_lab[:, :, 2] * saturation_mult
    
    # 3. APPLY ZONE TINTS directly
    # Weight by luminance zones (smooth gaussian transitions)
    def zone_weight(L_val, center, width):
        return np.exp(-0.5 * ((L_val - center) / width) ** 2)
    
    # Zone centers
    shadow_w = zone_weight(matched_l, 15, 18)
    lowmid_w = zone_weight(matched_l, 35, 12)
    midtone_w = zone_weight(matched_l, 50, 15)
    highmid_w = zone_weight(matched_l, 70, 12)
    highlight_w = zone_weight(matched_l, 90, 15)
    
    # Normalize weights
    total_w = shadow_w + lowmid_w + midtone_w + highmid_w + highlight_w + 1e-6
    
    # Get codebook values (with defaults)
    shadow = color_codebook.get('shadow', {'a': 0, 'b': 0})
    midtone = color_codebook.get('midtone', {'a': 0, 'b': 0})
    highlight = color_codebook.get('highlight', {'a': 0, 'b': 0})
    
    # Interpolate between zones
    tint_a = (
        shadow_w * shadow['a'] +
        lowmid_w * (shadow['a'] * 0.7 + midtone['a'] * 0.3) +
        midtone_w * midtone['a'] +
        highmid_w * (midtone['a'] * 0.7 + highlight['a'] * 0.3) +
        highlight_w * highlight['a']
    ) / total_w
    
    tint_b = (
        shadow_w * shadow['b'] +
        lowmid_w * (shadow['b'] * 0.7 + midtone['b'] * 0.3) +
        midtone_w * midtone['b'] +
        highmid_w * (midtone['b'] * 0.7 + highlight['b'] * 0.3) +
        highlight_w * highlight['b']
    ) / total_w
    
    # Apply tints
    graded_a = src_a + tint_a
    graded_b = src_b + tint_b
    
    # Clip to valid range
    graded_a = np.clip(graded_a, -128, 127)
    graded_b = np.clip(graded_b, -128, 127)
    
    # Combine
    graded_lab = np.zeros_like(src_lab)
    graded_lab[:, :, 0] = matched_l
    graded_lab[:, :, 1] = graded_a
    graded_lab[:, :, 2] = graded_b
    
    # Convert back to RGB
    graded_rgb = color.lab2rgb(graded_lab)
    graded_rgb = np.clip(graded_rgb, 0, 1).astype(np.float32)
    
    return graded_rgb


# MATRIX COLOR CODEBOOK - Hardcoded for quick test
# Matrix is famously "Olive Green" (Green + Yellow)
# Lab a: negative=green, Lab b: positive=yellow
MATRIX_CODEBOOK = {
    "shadow": {"a": -12, "b": 5},      # Deep olive shadows
    "midtone": {"a": -20, "b": 8},     # Strong sickly green midtones
    "highlight": {"a": -8, "b": 2},    # Subtle green in highlights
}


# Initialize modules
semantic_guide = SemanticGuide()
color_scientist = ColorScientist()
palette_extractor = PaletteExtractor()
color_code_extractor = ColorCodeExtractor()

@app.post("/api/analyze")
async def analyze_images(
    reference: UploadFile = File(...), 
    source: UploadFile = File(...),
    skip_ai: bool = False
):
    """
    Hybrid Luma-Chroma Analysis Pipeline (Compositional update):
    1. COMPOSITION: LLM deconstructs reference into semantic color regions.
    2. LUMA: Zone-based matching.
    3. CHROMA: Codebook-based tinting or Palette TPS.
    """
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    ref_path = os.path.join(session_dir, "ref.jpg")
    src_path = os.path.join(session_dir, "src.jpg")
    # proxy_path is no longer strictly needed for generation but we keep for backward compatibility in session
    proxy_path = os.path.join(session_dir, "v1_baseline.png")

    try:
        # Save uploaded files
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        with open(src_path, "wb") as f:
            shutil.copyfileobj(source.file, f)

        # Load images
        ref_img = load_image(ref_path)
        src_img = load_image(src_path)
        
        print(f"\n{'='*50}")
        print(f"Session {session_id}: HYBRID LUMA-CHROMA PIPELINE")
        print(f"{'='*50}")
        
        # ============================================
        # STEP 0: SEMANTIC COMPOSITION & ANALYSIS
        # ============================================
        print(f"\n[0/5] SEMANTIC ANALYSIS: Deconstructing image composition...")
        
        # 1. Analyze Source Image (Basic Zone Extraction)
        # We don't need LLM for source, just simple luma zones
        print("  Analyzing Source Image zones...")
        src_codebook = color_code_extractor.extract_codebook(src_img, {})
        
        # 2. Analyze Reference Image (Deep Semantic Extraction)
        # Use LLM to understand the *intent* of the reference colors
        print("  Analyzing Reference Image with Semantic Guide...")
        ref_composition = semantic_guide.analyze_composition(ref_img)
        ref_codebook = color_code_extractor.extract_codebook(ref_img, ref_composition)
        
        # 3. Get Global Parameters (Luma, Saturation)
        semantic_hints = semantic_guide.analyze_pair(src_img, ref_img)
        luma_strength = semantic_hints.get('luma_strength', 0.6)
        
        print(f"  ✓ Source Zones: {list(src_codebook.keys())}")
        print(f"  ✓ Target Zones: {list(ref_codebook.keys())}")
        
        
        # ============================================
        # STEP 1: LUMA ANCHOR (Zone-Based)
        # ============================================
        print(f"\n[1/5] LUMA ANCHOR: Zone-based histogram matching...")
        luma_mapper = compute_zoned_luma_mapper(
            src_img, ref_img, 
            strength=luma_strength
        )
        
        
        # ============================================
        # STEP 2: CHROMA WARP (Compositional Matching)
        # ============================================
        print(f"\n[2/5] CHROMA WARP: Mapping Source Zones -> Target Zones...")
        
        pins_source = []
        pins_target = []
        pin_labels = []
        
        # Define approximate L centers for zones to enable 3D warping
        # This allows gradients (e.g. shadows=green, highlights=warm)
        zone_centers = {
            'shadow': 15,    # Deep shadows
            'midtone': 50,   # Mid-gray
            'highlight': 85  # Bright highlights
        }
        
        # 1. Match primary zones (Shadow, Midtone, Highlight)
        for zone in ['shadow', 'midtone', 'highlight']:
            if zone in src_codebook and zone in ref_codebook:
                s = src_codebook[zone]
                t = ref_codebook[zone]
                
                # Create 3D pin: [L, a, b]
                l_val = zone_centers.get(zone, 50)
                pins_source.append([l_val, s['a'], s['b']])
                pins_target.append([l_val, t['a'], t['b']])
                
                pin_labels.append(zone)
                print(f"  Link: {zone.upper()} (L={l_val}, {s['a']:.1f},{s['b']:.1f}) -> ({t['a']:.1f},{t['b']:.1f})")

        # 2. Add Global Tint anchor (Neutral -> Global Tint)
        # Add anchors at multiple L levels to ensure global coverage
        if 'global' in ref_codebook:
             t = ref_codebook['global']
             for l_val in [10, 50, 90]:
                 pins_source.append([l_val, 0, 0]) # Neutral gray at L
                 pins_target.append([l_val, t['a'], t['b']]) # Tinted gray at L
             pin_labels.append("global_cast")
             print(f"  Link: GLOBAL CAST (Neutrals) -> ({t['a']:.1f},{t['b']:.1f})")

        # 3. Fit TPS
        if len(pins_source) >= 3:
            chroma_tps = ThinPlateSpline(smoothing=0.1) # Smooth warp
            chroma_tps.fit(np.array(pins_source), np.array(pins_target))
            print("  ✓ TPS fitted successfully")
        else:
            print("  ⚠ Not enough pins for TPS, using identity")
            chroma_tps = None 
            
            
        # ============================================
        # STEP 3: BASELINE (Legacy Support)
        # ============================================
        print(f"\n[3/5] BASELINE: Generated internal structures")
        proxy_path = os.path.join(session_dir, "v1_baseline.png")
        shutil.copy(src_path, proxy_path)
        proxy_status = {"used": False}
        
        # Placeholder for palette analysis to keep data structure compliant
        palette_analysis = {'palette': []}
        
        # ============================================
        # STEP 5: GENERATE VISUALIZATION DATA
        # ============================================
        print(f"\\n[5/5] Generating visualization lattice...")
        viz_size = 10
        viz_lattice_rgb = generate_identity_lattice(size=viz_size)
        
        # Apply the hybrid grade to the visualization lattice
        # Reshape lattice for Lab conversion
        lattice_img = viz_lattice_rgb.reshape(viz_size, viz_size * viz_size, 3)
        
        # For viz, we use identity luma mapping (just show chroma warp)
        identity_luma = interp1d([0, 100], [0, 100], kind='linear', 
                                 bounds_error=False, fill_value=(0, 100))
        
        warped_lattice_img = apply_hybrid_grade(lattice_img, identity_luma, chroma_tps)
        warped_viz = warped_lattice_img.reshape(-1, 3)
        
        # Prepare response with proxy status
        response_data = {
            "session_id": session_id,
            "original_points": viz_lattice_rgb.tolist(),
            "warped_points": warped_viz.tolist(),
            "pins_source": [],
            "pins_target": [],
            "proxy_status": proxy_status
        }
        
        # Save session for later use
        # Generate V1 preview for history
        session_data = {
            "ref_path": ref_path,
            "src_path": src_path,
            "proxy_path": proxy_path,
            "luma_mapper": luma_mapper,
            "chroma_tps": chroma_tps,
            "pins_src": pins_source,
            "pins_tgt": pins_target,
            "pin_labels": pin_labels,
            "semantic_hints": semantic_hints,
            "composition": ref_composition,
            "codebook": ref_codebook,
            "src_codebook": src_codebook,
            "v1_preview": None  # Will be set by preview endpoint
        }
        PipelineState.sessions[session_id] = session_data
        
        print(f"\n✓ Analysis complete. Session: {session_id}")
        print(f"{'='*50}\n")
        
        return response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/api/preview/{session_id}")
async def preview_graded(session_id: str):
    """
    Apply the FULL hybrid grade (Luma + Chroma) to the source image.
    Returns base64 encoded JPEG preview.
    """
    if session_id not in PipelineState.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = PipelineState.sessions[session_id]
    luma_mapper = session['luma_mapper']
    chroma_tps = session['chroma_tps']
    src_path = session['src_path']
    
    # Load and apply hybrid grade
    src_img = load_image(src_path)
    graded_img = apply_hybrid_grade(src_img, luma_mapper, chroma_tps)
    
    # Convert to uint8 and encode
    graded_uint8 = (graded_img * 255).astype(np.uint8)
    graded_bgr = cv2.cvtColor(graded_uint8, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', graded_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    import base64
    b64_str = base64.b64encode(buffer).decode('utf-8')
    preview_data = f"data:image/jpeg;base64,{b64_str}"
    
    # Store V1 preview for history
    if session.get('v1_preview') is None:
        session['v1_preview'] = preview_data
    
    return {"preview": preview_data}


@app.get("/api/preview-direct/{session_id}")
async def preview_direct_grade(session_id: str):
    """
    EXPERIMENTAL: Apply direct color grade using codebook.
    
    This bypasses TPS and applies hardcoded Matrix Lab values directly.
    Used to test the compositional color matching approach.
    """
    if session_id not in PipelineState.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = PipelineState.sessions[session_id]
    luma_mapper = session['luma_mapper']
    src_path = session['src_path']
    
    # Load source
    src_img = load_image(src_path)
    
    # Apply direct grade with Matrix codebook
    graded_img = apply_direct_color_grade(
        src_img, 
        luma_mapper, 
        MATRIX_CODEBOOK,
        saturation_mult=0.85
    )
    
    # Encode
    graded_uint8 = (graded_img * 255).astype(np.uint8)
    graded_bgr = cv2.cvtColor(graded_uint8, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', graded_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    import base64
    b64_str = base64.b64encode(buffer).decode('utf-8')
    preview_data = f"data:image/jpeg;base64,{b64_str}"
    
    print(f"[DIRECT GRADE] Applied Matrix codebook to session {session_id}")
    print(f"  Codebook: {MATRIX_CODEBOOK}")
    
    return {"preview": preview_data, "codebook": MATRIX_CODEBOOK}


@app.get("/api/debug/{session_id}")
async def get_debug_images(session_id: str):
    """
    Return reference and proxy images side by side for debugging.
    """
    if session_id not in PipelineState.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = PipelineState.sessions[session_id]
    ref_path = session['ref_path']
    proxy_path = session.get('proxy_path', ref_path.replace('ref.jpg', 'proxy.png'))
    
    import base64
    
    # Load and encode reference
    ref_img = load_image(ref_path)
    ref_uint8 = (ref_img * 255).astype(np.uint8)
    ref_bgr = cv2.cvtColor(ref_uint8, cv2.COLOR_RGB2BGR)
    _, ref_buffer = cv2.imencode('.jpg', ref_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ref_b64 = base64.b64encode(ref_buffer).decode('utf-8')
    
    # Load and encode proxy
    if os.path.exists(proxy_path):
        proxy_img = load_image(proxy_path)
        # Resize proxy to match reference for proper comparison
        if proxy_img.shape[:2] != ref_img.shape[:2]:
            proxy_img = cv2.resize(proxy_img, (ref_img.shape[1], ref_img.shape[0]))
        proxy_uint8 = (proxy_img * 255).astype(np.uint8)
        proxy_bgr = cv2.cvtColor(proxy_uint8, cv2.COLOR_RGB2BGR)
        _, proxy_buffer = cv2.imencode('.jpg', proxy_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        proxy_b64 = base64.b64encode(proxy_buffer).decode('utf-8')
    else:
        proxy_b64 = ref_b64  # Fallback to reference if no proxy
    
    return {
        "reference": f"data:image/jpeg;base64,{ref_b64}",
        "proxy": f"data:image/jpeg;base64,{proxy_b64}",
        "ref_size": list(ref_img.shape[:2]),
        "proxy_exists": os.path.exists(proxy_path)
    }


@app.get("/api/download/{session_id}")
async def download_lut(session_id: str):
    """
    Bake the hybrid grade into a 33x33x33 .cube LUT file.
    The LUT encodes both Luma anchor AND Chroma warp.
    """
    if session_id not in PipelineState.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = PipelineState.sessions[session_id]
    luma_mapper = session['luma_mapper']
    chroma_tps = session['chroma_tps']
    
    lut_path = os.path.join(UPLOAD_DIR, session_id, "Grade.cube")
    
    # Generate 33x33x33 identity lattice
    size = 33
    lattice_rgb = generate_identity_lattice(size=size)
    
    # Reshape for Lab processing
    lattice_img = lattice_rgb.reshape(size, size * size, 3)
    
    # Apply full hybrid grade
    graded_lattice_img = apply_hybrid_grade(lattice_img, luma_mapper, chroma_tps)
    graded_lattice = graded_lattice_img.reshape(-1, 3)
    
    # Export
    export_to_cube(graded_lattice, lut_path, size=size)
    
    return FileResponse(lut_path, media_type='application/octet-stream', filename="HCGE_Grade.cube")


@app.post("/api/refine/{session_id}")
async def refine_with_ai(session_id: str, max_rounds: int = 4):
    """
    AI Color Scientist refinement loop.
    Iteratively improves the grade by comparing to reference.
    
    Returns refinement history with feedback from each round.
    """
    if session_id not in PipelineState.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = PipelineState.sessions[session_id]
    ref_path = session['ref_path']
    src_path = session['src_path']
    luma_mapper = session['luma_mapper']
    chroma_tps = session['chroma_tps']
    
    # Get current adjustment params (or initialize)
    current_params = session.get('ai_adjustments', {
        'saturation_boost': 1.2,
        'luma_strength': 0.6,
        'blue_shift': 0,
        'green_shift': 0,
        'shadow_adjust': 0,
        'highlight_adjust': 0
    })
    
    # Load images
    ref_img = load_image(ref_path)
    src_img = load_image(src_path)
    
    # Initialize Color Scientist
    scientist = ColorScientist()
    
    refinement_history = []
    
    # Extract semantic context if available
    semantic_context = ""
    if 'semantic_hints' in session:
        hints = session['semantic_hints']
        description = hints.get('description', '')
        dominant_hue = hints.get('dominant_hue_name', '')
        
        parts = []
        if description:
            parts.append(f"Description: {description}")
        if dominant_hue:
            parts.append(f"Dominant Hue: {dominant_hue}")
            
        semantic_context = " | ".join(parts)
    
    print(f"\n{'='*50}")
    print(f"AI COLOR SCIENTIST REFINEMENT")
    print(f"Session: {session_id}")
    print(f"Max rounds: {max_rounds}")
    print(f"Context: {semantic_context}")
    print(f"{'='*50}")
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n[Round {round_num}/{max_rounds}]")
        
        # Generate current graded image with current params
        graded_img = apply_hybrid_grade(
            src_img, 
            luma_mapper, 
            chroma_tps, 
            saturation_boost=current_params.get('saturation_boost', 1.2)
        )
        
        # Apply additional AI adjustments if any
        if any(current_params.get(k, 0) != 0 for k in ['blue_shift', 'green_shift', 'shadow_adjust', 'highlight_adjust', 
                                                      'shadow_tint_a', 'shadow_tint_b', 'midtone_tint_a', 'midtone_tint_b',
                                                      'highlight_tint_a', 'highlight_tint_b']):
            graded_lab = color.rgb2lab(graded_img)
            graded_lab = apply_ai_adjustments(graded_lab, current_params)
            graded_img = color.lab2rgb(graded_lab).astype(np.float32)
            graded_img = np.clip(graded_img, 0, 1)
        
        # Ask AI to analyze and suggest
        new_params, feedback, satisfied = scientist.analyze_and_suggest(
            ref_img, graded_img, current_params, semantic_context=semantic_context
        )
        
        refinement_history.append({
            'round': round_num,
            'params': new_params.copy(),
            'feedback': feedback,
            'satisfied': satisfied
        })
        
        if satisfied:
            print(f"\n✓ AI is satisfied with the grade!")
            break
        
        current_params = new_params
    
    # Store final params in session
    session['ai_adjustments'] = current_params
    
    # Generate final preview
    final_graded = apply_hybrid_grade(
        src_img, luma_mapper, chroma_tps,
        saturation_boost=current_params.get('saturation_boost', 1.2)
    )
    if any(current_params.get(k, 0) != 0 for k in ['blue_shift', 'green_shift', 'shadow_adjust', 'highlight_adjust',
                                                  'shadow_tint_a', 'shadow_tint_b', 'midtone_tint_a', 'midtone_tint_b',
                                                  'highlight_tint_a', 'highlight_tint_b']):
        final_lab = color.rgb2lab(final_graded)
        final_lab = apply_ai_adjustments(final_lab, current_params)
        final_graded = color.lab2rgb(final_lab).astype(np.float32)
        final_graded = np.clip(final_graded, 0, 1)
    
    # Encode preview
    graded_uint8 = (final_graded * 255).astype(np.uint8)
    graded_bgr = cv2.cvtColor(graded_uint8, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', graded_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    preview_b64 = base64.b64encode(buffer).decode('utf-8')
    
    print(f"\n✓ Refinement complete after {len(refinement_history)} rounds")
    print(f"{'='*50}\n")
    
    return {
        'session_id': session_id,
        'rounds_completed': len(refinement_history),
        'final_params': current_params,
        'history': refinement_history,
        'preview': f"data:image/jpeg;base64,{preview_b64}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
