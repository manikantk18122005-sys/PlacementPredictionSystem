"""
DBSCAN Clustering on Student Placement Dataset
Detects dense student performance clusters and isolates anomaly/outlier students.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import (
    split_X_data,
    identity_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

def run_dbscan_placement(eps=1.8, min_samples=10, sample_size=1000):
    df = load_data()
    print(f"Loaded {df.shape[0]} placement records.")

    # Sample for efficient density analysis
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()

    X = split_X_data(
        df_sample,
        drop_columns=['StudentID', 'PlacementStatus', 'IsAnomaly', 'Salary Package']
    )

    numerical_features, categorical_features = identity_features(X)
    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']

    X, _, _ = handle_missing_values(X, X, numerical_features)
    X, _, _ = standardize_data(X, X, numerical_features)
    X, _, _ = one_hot_encode_data(X, X, one_hot_features)
    X, _, _ = ordinal_encode_data(X, X, ordinal_features)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nDBSCAN Clustering Results (eps={eps}, min_samples={min_samples}):")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise/outlier points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

    # Visualize with PCA 2D projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap='tab10',
        edgecolors='k', alpha=0.7, s=40
    )
    plt.title(f"DBSCAN Density Clustering on Placement Data (Clusters: {n_clusters}, Outliers: {n_noise})")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return dbscan, labels

if __name__ == "__main__":
    run_dbscan_placement()
