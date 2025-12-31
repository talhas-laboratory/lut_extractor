
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
