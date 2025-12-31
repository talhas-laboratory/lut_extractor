
import numpy as np
from skimage.exposure import match_histograms

def match_cumulative_cdf(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Apply global histogram matching (CDF) to align the source image's 
    tonality to the reference. This is primarily for the Luma channel.

    Args:
        source (np.ndarray): Source image array (or channel).
        reference (np.ndarray): Reference image array (or channel).

    Returns:
        np.ndarray: The matched source image.
    """
    # skimage.exposure.match_histograms handles multi-channel and single-channel
    # accurately using CDF matching.
    
    # Ensure inputs are float32
    if source.dtype != np.float32:
        source = source.astype(np.float32)
    if reference.dtype != np.float32:
        reference = reference.astype(np.float32)

    # match_histograms returns float64 usually, cast back to float32
    matched = match_histograms(source, reference, channel_axis=-1 if source.ndim == 3 else None)
    
    return matched.astype(np.float32)
