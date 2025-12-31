# Master Blueprint: Hybrid AI-Math Color Grading Engine (HCGE)

## I. Product Philosophy

The **HCGE** is a high-fidelity style transfer engine designed for professional creators. It bridges the gap between the intuition of Generative AI and the precision of Color Science. Unlike "blind" LUTs or "dumb" histogram matches, HCGE understands the **intent** of a color grade by reconstructing the source reality before the stylization.

### The "Context-Aware" Heist

Most matching tools fail because they match pixels, not concepts. HCGE matches **transformations**. By seeing how an AI "de-graded" proxy relates to the final cinematic still, the engine identifies exactly how colors were shifted, then applies that "bending" to the user's footage while anchoring the exposure to the user's actual camera data.

---

## II. System Architecture

The pipeline follows a strict **3-Step Hybrid Logic**:

1. **AI De-Grading (The Decoder):** Generates a "Neutral Proxy" of the cinematic reference.
2. **Luma-Chroma Split (The Engine):** Separates Light (Physics) from Color (Art).
3. **Physical Warp (The Result):** Uses Thin Plate Spline (TPS) math to bend the color spectrum like a rigid metal sheet, ensuring buttery smooth transitions.

---

## III. Implementation Specification

### 1. Environment & Dependencies

* **Python 3.10+**
* **Core:** `NumPy` (Tensor math), `SciPy` (RBF Interpolation), `OpenCV` (IO & Conversions)
* **Matching:** `Scikit-image` (Histogram/CDF logic)
* **AI:** `google-generativeai` (Gemini 1.5 Pro for Proxy generation)

### 2. The Color Space "Range Law"

To prevent math errors, the backend must adhere to **Normalized Float32** throughout:

* **Input/Output:** RGB scaled to .
* **CIELAB Range:**
* **L:** 
* **a:** 
* **b:** 



### 3. AI Proxy Prompting (Structure Preservation)

When generating the `proxy_neutral.png`, use this prompt to ensure pixel-perfect alignment:

> *"Act as a DIT (Digital Imaging Technician). Neutralize all color grading from this image. Restore natural skin tones (CIE D65) and Rec.709 white balance. **STRICT CONSTRAINT:** Do not crop, zoom, or alter the composition. Every pixel in the output must align perfectly with the input."*

### 4. Mathematical Foundations

#### The Luma Anchor (L-Channel)

Apply **Direct Cumulative Distribution Function (CDF)** matching between the User Source and the Reference:


* *Result:* This steals the movie's contrast but prevents the user's footage from overexposing.

#### The Chroma Warp (TPS Algorithm)

The "bending" of the color cloud minimizes the **Bending Energy Functional**:


* **Implementation:** Use `scipy.interpolate.RBFInterpolator` with the `thin_plate_spline` kernel.
* **Pins:** Use 8 Cube Corners (Boundary Pins) + 1 Semantic Skin Pin + 24 Statistical Centroids (K-Means).

---

## IV. The .cube Factory (Lattice Baking)

The final transformation is projected onto a  identity lattice:

1. **Generate Identity:**  RGB points .
2. **Warp:** Pass the lattice through the TPS model calculated from the Proxy-to-Reference relationship.
3. **Clip:** Force `np.clip(final_rgb, 0.0, 1.0)` to ensure high-nit highlights don't "flip" into unintended colors.
4. **Export:** Standard Iridas `.cube` format.

---

## V. AGENTS.md (Instructions for AI Builders)

### Core Directives

1. **Strict Typing:** Use `numpy.float32`. Never downcast to `uint8` during math.
2. **Vectorization:** Use NumPy broadcasting for the RBF calculations. Do not iterate through the 35,937 lattice nodes using loops.
3. **Library Guardrail:** Use `scipy.interpolate.RBFInterpolator` (Modern) instead of `scipy.interpolate.Rbf` (Legacy).

### Execution Flow

1. `convert_to_lab(image)`
2. `compute_popularity_pins(proxy_img, ref_img, n=24)`
3. `solve_tps_coefficients(source_pins, target_pins, smoothing=0.1)`
4. `apply_luma_anchor(user_L, ref_L)`
5. `bake_lattice_to_cube(output_path)`
