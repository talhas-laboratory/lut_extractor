
import cv2
import numpy as np
from skimage import color

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and convert to RGB float32 [0, 1].

    Args:
        path (str): File path.

    Returns:
        np.ndarray: float32 RGB image (H, W, 3) in range [0, 1].
    """
    # OpenCV loads as BGR uint8
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to float32 [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    return img_float

def save_image(image: np.ndarray, path: str) -> None:
    """
    Save a float32 RGB image to disk.

    Args:
        image (np.ndarray): float32 RGB image [0, 1].
        path (str): Target path.
    """
    # Clip to valid range
    img_clipped = np.clip(image, 0.0, 1.0)
    
    # Convert to uint8
    img_uint8 = (img_clipped * 255.0).astype(np.uint8)
    
    # RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, img_bgr)

def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB float32 [0, 1] to CIELAB.
    L: [0, 100], a: [-128, 127], b: [-128, 127] roughly.
    """
    return color.rgb2lab(image)

def lab_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert CIELAB to RGB float32 [0, 1].
    """
    return color.lab2rgb(image)
