# tests/test_clustering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load test data (use a small sample or mock RFM for tests)
rfm_test = pd.read_csv('data/rfm_test.csv')

def test_scaling():
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_test[['Recency', 'Frequency', 'Monetary']])
    assert rfm_scaled.shape[1] == 3, "Scaled RFM should have 3 columns"
    assert np.allclose(rfm_scaled.mean(axis=0), 0, atol=1e-6), "Scaled mean should be ~0"
    assert np.allclose(rfm_scaled.std(axis=0), 1, atol=1e-6), "Scaled std should be ~1"

def test_elbow_computation():
    inertia, k_range = [], range(1, 4)  # Small range for test
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_test[['Recency', 'Frequency', 'Monetary']])  # Unscaled for simple test
        inertia.append(kmeans.inertia_)
    assert len(inertia) == len(k_range), "Inertia list should match k_range length"
    assert all(inertia[i] >= inertia[i+1] for i in range(len(inertia)-1)), "Inertia should decrease with k"

def test_silhouette_computation():
    sil_scores = []
    for k in range(2, 4):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(rfm_test[['Recency', 'Frequency', 'Monetary']])
        sil_scores.append(silhouette_score(rfm_test[['Recency', 'Frequency', 'Monetary']], clusters))
    assert len(sil_scores) == 2, "Should compute for k=2 to 3"

def test_fit_and_assign():
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(rfm_test[['Recency', 'Frequency', 'Monetary']])
    assert len(clusters) == len(rfm_test), "Clusters should match RFM rows"
    assert len(set(clusters)) == 2, "Expected 2 unique clusters"

# Run: pytest tests/test_clustering.py