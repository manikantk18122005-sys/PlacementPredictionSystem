from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

def create_data():
    X, y = make_moons(n_samples=300, noise=0.08, random_state=42)
    return X, y

def train_kmeans(X):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def train_dbscan(X, eps=0.2, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return dbscan, labels

def display_results(X, kmeans_labels, dbscan_labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # K-Means plot
    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k', s=35)
    axes[0].set_title("K-Means Clustering (Fails on Non-Spherical Moons)")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # DBSCAN plot
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='plasma', edgecolors='k', s=35)
    axes[1].set_title("DBSCAN Clustering (Correctly Captures Arbitrary Shapes)")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("K-Means vs DBSCAN on Non-Spherical Moons Dataset", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    X, _ = create_data()
    print("Moons data shape:", X.shape)
    
    kmeans, km_labels = train_kmeans(X)
    print("K-Means clustering completed.")
    
    dbscan, db_labels = train_dbscan(X, eps=0.2, min_samples=5)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points detected.")
    
    display_results(X, km_labels, db_labels)

if __name__ == "__main__":
    main()
