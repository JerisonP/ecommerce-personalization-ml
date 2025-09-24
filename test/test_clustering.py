def test_scaling(rfm_scaled):
  assert rfm_scaled.shape[1] == 3, "Scaled RFM should have 3 columns"
  assert abs(rfm_scaled.mean()) < 1e-6, "Scaled data should have mean ~0"

def test_clusters(rfm):
  assert 'Cluster' in rfm.columns, "Cluster column missing"
  assert len(rfm['Cluster'].unique()) == 4, "Expected 4 unique clusters"