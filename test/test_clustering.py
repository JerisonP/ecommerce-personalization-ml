# test/test_clustering.py
import sys
sys.path.append('..')  # Add root to path for imports

import pandas as pd
import numpy as np
from my_library.clustering import scale_rfm, compute_elbow, compute_silhouette, fit_and_assign_clusters  # Import from library

# Load test data (your rfm_test.csv; fallback to mock if file missing)
try:
    rfm_test = pd.read_csv('../data/rfm_test.csv')
except FileNotFoundError:
    # Mock fallback
    rfm_test = pd.DataFrame({
        'Recency': [10, 20, 30],
        'Frequency': [5, 10, 15],
        'Monetary': [100, 200, 300]
    })

def test_scaling():
    rfm_scaled = scale_rfm(rfm_test)
    assert rfm_scaled.shape[1] == 3, "Scaled RFM should have 3 columns"
    assert np.allclose(rfm_scaled.mean(axis=0), 0, atol=1e-6), "Scaled mean ~0"
    assert np.allclose(rfm_scaled.std(axis=0), 1, atol=1e-6), "Scaled std ~1"

def test_compute_elbow():
    rfm_scaled = scale_rfm(rfm_test)
    inertia, k_range = compute_elbow(rfm_scaled, range(1, 4))  # Small range for test
    assert len(inertia) == len(k_range), "Inertia list matches k_range"
    assert all(inertia[i] >= inertia[i+1] for i in range(len(inertia)-1)), "Inertia decreases with k"

def test_compute_silhouette():
    rfm_scaled = scale_rfm(rfm_test)
    sil_scores, k_range = compute_silhouette(rfm_scaled, range(2, 4))
    assert len(sil_scores) == len(k_range), "Sil scores match k_range"

def test_fit_and_assign_clusters():
    rfm_scaled = scale_rfm(rfm_test)
    rfm_copy, kmeans = fit_and_assign_clusters(rfm_test.copy(), rfm_scaled, k=2)
    assert 'Cluster' in rfm_copy.columns, "Cluster column added"
    assert len(rfm_copy['Cluster'].unique()) == 2, "Expected 2 unique clusters"