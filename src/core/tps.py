
import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Optional

class ThinPlateSpline:
    """
    Implements Thin Plate Spline (TPS) warping using Radial Basis Functions.
    This class is responsibly for learning the transformation between two sets of
    3D points (RGB colors) and applying that transformation to a new set of points.
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Initialize the TPS model.

        Args:
            smoothing (float): Smoothing parameter (epsilon/lambda) to prevent overfitting
                             and reduce banding. Default is 0.1 as per guide.
        """
        self.smoothing = smoothing
        self.rbf: Optional[RBFInterpolator] = None

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> None:
        """
        Fit the TPS model to the source and target control points.

        Args:
            source_points (np.ndarray): Shape (N, 3) - Input colors (Proxy/Source).
            target_points (np.ndarray): Shape (N, 3) - Output colors (Reference).
        """
        if source_points.shape != target_points.shape:
            raise ValueError("Source and target points must have the same shape.")
        
        if source_points.shape[1] != 3:
            raise ValueError("Points must be 3D coordinates (RGB/LAB).")

        # The kernel 'thin_plate_spline' is standard for this type of warp.
        # RBFInterpolator efficiently handles N-D to M-D interpolation.
        self.rbf = RBFInterpolator(
            source_points, 
            target_points, 
            kernel='thin_plate_spline', 
            smoothing=self.smoothing
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the learned transformation to a set of points.

        Args:
            points (np.ndarray): Shape (M, 3) - Points to transform (e.g., Lattice).

        Returns:
            np.ndarray: Transformed points, same shape as input.
        """
        if self.rbf is None:
            raise RuntimeError("TPS model has not been fitted yet.")
        
        if points.shape[1] != 3:
            raise ValueError("Points must be 3D coordinates.")

        return self.rbf(points)
