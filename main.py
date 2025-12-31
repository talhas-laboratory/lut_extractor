
import argparse
import sys
import os
import numpy as np
from src.io.loader import load_image, save_image
from src.ai.proxy import GeminiProxyGenerator
from src.core.pins import compute_pins
from src.core.tps import ThinPlateSpline
from src.core.matcher import match_cumulative_cdf
from src.io.baker import generate_identity_lattice, export_to_cube

def main():
    parser = argparse.ArgumentParser(description="HCGE: Hybrid AI-Math Color Grading Engine")
    parser.add_argument("--ref", required=True, help="Path to Reference Image (The 'Look')")
    parser.add_argument("--source", required=True, help="Path to Source Image (The User's Log/Flat footage or frame)")
    parser.add_argument("--output", default="output.cube", help="Path to export the .cube LUT")
    parser.add_argument("--proxy_path", default="proxy_neutral.png", help="Path to save/load the AI generated proxy")
    parser.add_argument("--skip_ai", action="store_true", help="Skip AI gen and use existing proxy_path")
    
    args = parser.parse_args()

    # 1. Load Images
    print(f"[1/5] Loading images...")
    try:
        ref_img = load_image(args.ref)
        source_img = load_image(args.source)
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.exit(1)

    # 2. Generate/Load Proxy
    # The 'Source' image is the User's image. 
    # WAIT. The Guide says: "By seeing how an AI 'de-graded' proxy relates to the final cinematic still..."
    # "1. AI De-Grading (The Decoder): Generates a 'Neutral Proxy' of the CINEMATIC REFERENCE."
    # So we generate the proxy FROM THE REFERENCE.
    # Ref (Graded) -> Proxy (Neutral).
    # Then we learn the transform Proxy -> Ref.
    # Then we apply that transform to the User's Source.
    
    print(f"[2/5] Handling AI Proxy...")
    if args.skip_ai:
        if not os.path.exists(args.proxy_path):
            print(f"Error: --skip_ai used but {args.proxy_path} does not exist.")
            sys.exit(1)
        print(f"Loading existing proxy from {args.proxy_path}...")
        proxy_img = load_image(args.proxy_path)
    else:
        print(f"Generating Neutral Proxy for Reference: {args.ref}...")
        generator = GeminiProxyGenerator()
        success = generator.generate_proxy(args.ref, args.proxy_path)
        
        # Check if file exists now (since generate_proxy might fail or stubbed)
        if success and os.path.exists(args.proxy_path):
             proxy_img = load_image(args.proxy_path)
        else:
            print("AI Generation failed or did not produce an image file.")
            print("CRITICAL: For this prototype, ensure you have a valid proxy or an API enabling image output.")
            print("Assuming manual intervention or mocked proxy for now.")
            # Verify if user has provided a proxy manually?
            if os.path.exists(args.proxy_path):
                print(f"Found {args.proxy_path}, using it.")
                proxy_img = load_image(args.proxy_path)
            else:
                print("No proxy available. Aborting.")
                sys.exit(1)

    # Resize proxy to match reference if needed (Gemini might resize?)
    # TPS/Pins calculation needs correspondence?
    # Actually, we calculate stats (Pins) so pixel-perfect alignment isn't STRICTLY required
    # for the CLUSTERING, but for the GUIDE "Every pixel... must align perfectly" implied
    # we might be using spatial correspondence? 
    # "Compute popularity pins(proxy_img, ref_img)" -> Uses K-Means on the color cloud.
    # So we don't need spatial alignment, just color stats.
    
    # 3. Calculate Pins & Learn TPS
    print(f"[3/5] Calculating Pins and Learning TPS...")
    # Pins are clusters from Proxy (Source Domain) -> Reference (Target Domain)
    # Wait. 
    # We want to map: [Neutral Color] -> [Graded Color].
    # So we want to learn: Model(Proxy) = Reference.
    # So Source Points = Proxy Colors, Target Points = Reference Colors?
    # BUT "Compute popularity pins... and solve TPS coefficients(source_pins, target_pins)"
    # How do we map the pins?
    # If we cluster Proxy and Reference separately, the cluster centers might not correspond 1:1 semantically!
    # (e.g. Cluster 1 in Proxy might be Sky, Cluster 1 in Ref might be Skin if sorting differs).
    
    # PROBLEM: K-Means on two different images independently does NOT guarantee correspondence.
    # SOLUTION: 
    # The guide says "Strict Constraint... Do not crop... align perfectly".
    # This implies we can use SPATIAL correspondence if pixels align.
    # OR we use the Proxy as the base for clustering, and then find the average color 
    # of those SAME PIXELS in the Reference image.
    # YES. "Spatial Semantic Correspondence".
    
    # Step 3a: Cluster the PROXY to find "Palette Centers" (e.g. Skin, Sky, Shadow).
    # Step 3b: For each cluster, find the mask of pixels in Proxy.
    # Step 3c: Calculate the average color of those SAME pixels in the Reference.
    # This ensures Cluster i (Proxy) maps to Cluster i (Ref).
    
    # Resizing check
    if proxy_img.shape != ref_img.shape:
        print(f"Warning: Proxy shape {proxy_img.shape} != Ref shape {ref_img.shape}. Resizing Proxy.")
        proxy_img = cv2.resize(proxy_img, (ref_img.shape[1], ref_img.shape[0]))
    
    # Flatten
    h, w, c = proxy_img.shape
    proxy_flat = proxy_img.reshape(-1, 3)
    ref_flat = ref_img.reshape(-1, 3)
    
    # Subsample for KMeans if huge
    from sklearn.cluster import KMeans
    
    print("  - Running K-Means on Proxy...")
    # Use logic similar to compute_pins but customized for correspondence
    n_clusters = 24
    sample_indices = np.random.choice(len(proxy_flat), min(10000, len(proxy_flat)), replace=False)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(proxy_flat[sample_indices])
    
    # Labels for all pixels (or a larger subset to match)
    # To be accurate, we can predict on the whole image or a larger sample.
    # Let's use a larger sample for the mapping step.
    mapping_indices = np.random.choice(len(proxy_flat), min(50000, len(proxy_flat)), replace=False)
    labels = kmeans.predict(proxy_flat[mapping_indices])
    
    source_pins = []
    target_pins = []
    
    print("  - Mapping Clusters...")
    for i in range(n_clusters):
        # Mask for this cluster
        mask = (labels == i)
        if np.sum(mask) == 0:
            continue
            
        # Get pixels
        p_res = proxy_flat[mapping_indices][mask]
        r_res = ref_flat[mapping_indices][mask]
        
        # Average color in Proxy
        avg_p = np.mean(p_res, axis=0)
        # Average color in Ref
        avg_r = np.mean(r_res, axis=0)
        
        source_pins.append(avg_p)
        target_pins.append(avg_r)
        
    source_pins = np.array(source_pins, dtype=np.float32)
    target_pins = np.array(target_pins, dtype=np.float32)
    
    # Add Boundaries
    boundaries = np.array([
        [0,0,0], [1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]
    ], dtype=np.float32)
    
    source_pins = np.vstack([source_pins, boundaries])
    target_pins = np.vstack([target_pins, boundaries])
    
    print(f"  - Fitting TPS with {len(source_pins)} pins...")
    tps = ThinPlateSpline(smoothing=0.1)
    tps.fit(source_pins, target_pins)
    
    # 4. Luma Anchor (CDF Match)
    # Guide: "4. apply_luma_anchor(user_L, ref_L)"
    # This implies we modify the user's Luma before the warp? 
    # Or is it a separate step?
    # "The pipeline... 2. Luma-Chroma Split... 3. Physical Warp"
    # "Luma... Apply Direct CDF matching between User Source and Reference"
    # Wait, if we bake a LUT, we can't easily bake a dynamic CDF match unless we freeze it based on the submitted frame.
    # We will assume we are baking the transform for THIS source frame basically, or a generic one.
    # Usually CDF match is highly image specific.
    # We will compute the CDF curve 0->1 mapping for Luma using the Source and Ref.
    # Then apply it to the Lattice as a pre-transform?
    # Or post-transform?
    # "Separates Light from Color... This steals the movie's contrast".
    # Logic:
    #   Input -> Split L, ab
    #   L_new = MatchCDF(L_in, L_ref)
    #   ab_new ... warp ...
    #   Recombine.
    
    # We can bake this into the lattice!
    # For every point in the lattice (r,g,b):
    #   1. Convert to Lab.
    #   2. Map L using the learned CDF curve.
    #   3. Warp ab using TPS? Or does TPS Warp everything?
    # Guide says: "Uses TPS math to bend the color spectrum... anchoring the exposure"
    # "Luma Anchor... Result: This steals the movie's contrast"
    # If TPS is learned on RGB (Proxy->Ref), it includes Luma and Chroma shifts.
    # But Proxy generation "Neutralizes color grading... restores natural skin tones".
    # So Proxy has "Neutral" contrast?
    # If we map Neutral -> Graded, we add the Grade's contrast.
    # But the User's Source might have DIFFERENT contrast than the Proxy.
    # So if we apply (Neutral->Ref) skew to (User Source), we might get weird results if User Source isn't "Neutral".
    # The Luma Anchor step seems to be: "Force the Luma statistics of the output to match the Reference".
    
    # Implementation:
    # 1. Generate Identity Lattice. # (Points in RGB)
    # 2. Warp Lattice with TPS. -> This transforms Ref-Style colors to... wait.
    #    TPS maps Proxy(Neutral) -> Ref(Graded).
    #    So Input (User) is treated as "Neutral-ish" and pushed to "Ref-ish".
    #    This puts the colors in the right place.
    # 3. BUT, "Luma Anchor":
    #    The Guide implies explicitly handling Luma.
    #    Maybe we shouldn't rely on TPS for Luma.
    
    # Let's follow "Execution Flow":
    # 1. convert_to_lab(image)
    # 2. compute_popularity_pins...
    # 3. solve_tps...
    # 4. apply_luma_anchor(user_L, ref_L)
    # 5. bake_lattice_to_cube
    
    # Doing step 4 on the LATTICE:
    # The Lattice represents sample points of the USER's image (0 to 1).
    # So `user_L` equivalent is the Lattice L values.
    # `ref_L` is the Reference Image L channel.
    # We need to map `lattice_L` s.t. it matches `ref_L`'s histogram?
    # NO. That would make every gray ramp look like the movie poster (contrast wise).
    # `match_histograms` matches the DISTRIBUTION.
    # If we map a flat gradient (Lattice) to match a specific image histogram, we get a posterized mess.
    # You calculate the mapping function Source_L -> Ref_L, and apply THAT function to the lattice.
    
    # Correct Logic for Luma Step:
    # 1. Calculate CDF mapping: F(l) = match(Source_L, Ref_L).
    # 2. Apply F to the Lattice L channel.
    
    # Wait, logic check:
    # If we do this, we ignore the TPS Luma shift?
    # Or do we mix them?
    # Guide: "Separates Light (Physics) from Color (Art)."
    # "Luma Anchor... Direct CDF... Chroma Warp (TPS)".
    # This implies:
    # New_L = CDF_Match(Old_L)
    # New_ab = TPS_Warp(Old_ab) ??
    # TPS is calculated on RGB pins usually.
    # If we run TPS on RGB, we get new RGB.
    # We can convert new RGB -> Lab -> replace L -> RGB.
    
    # Let's try:
    # 1. Warp Lattice RGB -> Lattice RGB' (via TPS).
    #    This gives us the "Color Warp" + some Luma shift from the Proxy->Ref relationship.
    # 2. Convert RGB' to Lab'.
    # 3. Calculate Independent Luma Mapping from Source_Img -> Ref_Img.
    #    Wait, if the Source_Img is "User's Log", and Ref is "Gritty Movie",
    #    Matching histogram applies the Contrast.
    # 4. Replace L' with L_matched.
    #    L_matched = Match(Original_Lattice_L, CDF_Function) ??
    #    No, match(Source_Image_L, Ref_Image_L) gives us a mapping.
    #    We assume the Lattice covers the range of Source Image.
    #    Ideally, we'd learn the transfer curve `T(l)` from Source->Ref images.
    #    Then apply `T(lattice_l)`.
    
    # Let's just implement the "Apply Luma Anchor" by using `match_histograms` reference logic on the Lattice?
    # No, `match_histograms` takes an image.
    # We need to learn the curve.
    # Scikit-image doesn't expose the curve easily.
    # For now, to suffice the "Prototype" and "Plan", I might skip complex curve extraction 
    # and just trust the TPS if we assume Proxy is good?
    # BUT Guide is specific. "Hybrid Logic".
    
    # Simpler approach for Luma Anchor in a LUT:
    # The LUT transforms input RGB. 
    # We want the output Luma to look like Ref Luma.
    # But a LUT doesn't know the input distribution.
    # We can only bake "User Source Frame" specific matching if the LUT is intended for THAT clip.
    # Yes, "Anchoring the exposure to the user's actual camera data."
    # So we use the provided `--source` image to calculate the stats.
    
    # Implementation:
    # 1. Match `source_img` Luma to `ref_img` Luma.
    #    `matched_source = match_cdf(source, ref)`
    # 2. Train a 1D interpolator (UnivariateSpline) mapping `source_L` -> `matched_source_L`.
    # 3. For the Lattice:
    #    `L_in` -> `L_out = interpolator(L_in)`.
    #    Combine with `ab` from TPS?
    #    If TPS operates on RGB, we convert Lattice->RGB->Lab -> `L_tps, a_tps, b_tps`.
    #    We use `a_tps, b_tps` (Chrominance from AI).
    #    We use `L_out` (Luminance from CDF).
    
    # Let's do that.
    
    # 5. Generate Lattice
    print(f"[4/5] Generating Lattice...")
    lattice_rgb = generate_identity_lattice(size=33)
    
    # 6. Apply TPS (Color Warp)
    print(f"  - Warping Lattice...")
    warped_lattice_rgb = tps.transform(lattice_rgb)
    warped_lattice_rgb = np.clip(warped_lattice_rgb, 0.0, 1.0)
    
    # 7. Apply Luma Anchor
    print(f"  - Applying Luma Anchor...")
    import cv2 
    # Convert Lattice to Lab
    # Note: scikit-image rgb2lab expects (H,W,3). Our lattice is (N,3). 
    # We reshape to (N, 1, 3) for the func.
    lattice_lab = color.rgb2lab(lattice_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    warped_lab = color.rgb2lab(warped_lattice_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    
    # Calculate Source Luma -> Ref Luma mapping
    # We use the matched_histograms on the actual images to find the 1D mapping.
    # Or just use `match_histograms` with the lattice IF we assume lattice represents uniform distribution?
    # No, we must learn from Source Image.
    
    # Extract L channels
    src_lab = color.rgb2lab(source_img)
    ref_lab = color.rgb2lab(ref_img)
    src_l = src_lab[:,:,0].flatten()
    ref_l = ref_lab[:,:,0].flatten()
    
    # Subsample for speed
    if len(src_l) > 10000:
        src_l = np.random.choice(src_l, 10000)
    if len(ref_l) > 10000:
        ref_l = np.random.choice(ref_l, 10000)
        
    # Sort to approximate CDF
    src_l_sorted = np.sort(src_l)
    ref_l_sorted = np.sort(ref_l)
    
    # We want to map Value v in Src -> Ref.
    # Mapping: Find percentile of v in Src, map to same percentile in Ref.
    # Interpolator
    from scipy.interpolate import interp1d
    # Create quantiles
    quantiles = np.linspace(0, 1, 100)
    src_q = np.quantile(src_l, quantiles)
    ref_q = np.quantile(ref_l, quantiles)
    
    # Constraint ends
    src_q[0], src_q[-1] = 0, 100
    ref_q[0], ref_q[-1] = 0, 100
    
    luma_mapper = interp1d(src_q, ref_q, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Apply to lattice
    # Lattice L (from Identity) is what we map.
    # Because the LUT input corresponds to the Source Image values.
    current_lattice_l = lattice_lab[:, 0]
    matched_l = luma_mapper(current_lattice_l)
    
    # Combine: Matched L + Warped Chroma (a, b)
    final_lab = np.zeros_like(warped_lab)
    final_lab[:, 0] = matched_l
    final_lab[:, 1] = warped_lab[:, 1]
    final_lab[:, 2] = warped_lab[:, 2]
    
    # Convert back to RGB
    final_rgb = color.lab2rgb(final_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    final_rgb = np.clip(final_rgb, 0.0, 1.0)
    
    # 8. Export
    print(f"[5/5] Exporting to {args.output}...")
    export_to_cube(final_rgb, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
