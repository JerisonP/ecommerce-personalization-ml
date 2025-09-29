# my_library/clustering.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def scale_rfm(rfm):
    """Scale RFM features (Recency, Frequency, Monetary)."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled

def compute_elbow(rfm_scaled, k_range=range(1, 11)):
    """Compute inertia for elbow plot."""
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)
    return inertia, k_range

def compute_silhouette(rfm_scaled, k_range=range(2, 11)):
    """Compute silhouette scores."""
    sil_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        sil_scores.append(silhouette_score(rfm_scaled, clusters))
    return sil_scores, k_range

def fit_and_assign_clusters(rfm, rfm_scaled, k=4):
    """Fit K-means and assign clusters to RFM."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans

def visualize_clusters(rfm):
    """Visualize 2D scatterplots of clusters."""
    import seaborn as sns
    sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='viridis')
    plt.title('Clusters: Recency vs Monetary')
    plt.show()

    sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=rfm, palette='viridis')
    plt.title('Clusters: Frequency vs Monetary')
    plt.show()