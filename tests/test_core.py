
import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.tps import ThinPlateSpline
from src.core.pins import compute_pins
from src.core.matcher import match_cumulative_cdf

def test_tps_identity():
    # If source == target, transform should be identity
    points = np.random.rand(10, 3).astype(np.float32)
    tps = ThinPlateSpline(smoothing=0.0) # No smoothing for exact fit
    tps.fit(points, points)
    
    transformed = tps.transform(points)
    assert np.allclose(points, transformed, atol=1e-5)

def test_pins_shape():
    img = np.random.rand(100, 100, 3).astype(np.float32)
    pins = compute_pins(img, n_clusters=10)
    # 10 clusters + 8 corners = 18 pins
    assert pins.shape == (18, 3)
    # Check if corners are present
    assert np.allclose(pins[-1], [0, 1, 1]) # Cyan corner

def test_matcher_shape():
    src = np.random.rand(100, 100, 3).astype(np.float32)
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    matched = match_cumulative_cdf(src, ref)
    assert matched.shape == src.shape
