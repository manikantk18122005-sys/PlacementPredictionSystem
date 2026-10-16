import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.data.load_data import load_data, BASE_DIR
from src.data.preprocess import (
    identity_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)


def find_optimal_k(X):
    wcss = []
    silhouette_scores = []

    for k in range(2, 11):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )
        labels = model.fit_predict(X)
        wcss.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    print("\nWCSS and Silhouette Values:")
    for k, (w, s) in enumerate(zip(wcss, silhouette_scores), start=2):
        print(f"K = {k}, WCSS = {w:.2f}, Silhouette = {s:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(range(2, 11), wcss, marker="o")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("WCSS")
    axes[0].set_title("Elbow Method for Optimal K")
    axes[0].set_xticks(range(2, 11))
    axes[0].grid(True)

    axes[1].plot(range(2, 11), silhouette_scores, marker="o", color="green")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xticks(range(2, 11))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    best_k = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nSuggested K (highest silhouette): {best_k}")

    return wcss, silhouette_scores, best_k


def create_model(k):
    return KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )


def train_model(model, X):
    model.fit(X)
    print("\nK-Means trained successfully!")
    return model


def evaluate_model(model, X, labels):
    silhouette = silhouette_score(X, labels)

    print("\nModel Evaluation")
    print("-" * 25)
    print(f"WCSS: {model.inertia_:.2f}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Silhouette Score: {silhouette:.4f}")

    return silhouette


def get_cluster_centers(model, scaler, numerical_features, X_columns):
    """
    Inverse-transform cluster centers back to original scale
    for the numerical features only.
    """
    import pandas as pd

    centers_df = pd.DataFrame(model.cluster_centers_, columns=X_columns)

    num_cols_present = [c for c in numerical_features if c in centers_df.columns]
    if scaler is not None and num_cols_present:
        centers_df[num_cols_present] = scaler.inverse_transform(
            centers_df[num_cols_present]
        )

    return centers_df


def display_clusters_pca(X, labels, model):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(model.cluster_centers_)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="viridis", s=30
    )
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        marker="X", s=200, color="red", label="Centroids"
    )
    plt.title("K-Means Clustering (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.show()


def main():
    # Load Dataset
    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    # Select Features — drop non-feature / target columns
    # (your preprocess.py has no split_X_data, so we drop columns directly)
    drop_columns = [
        "StudentID",
        "PlacementStatus",
        "Salary Package",
        "IsAnomaly"
    ]
    X = df.drop(columns=[c for c in drop_columns if c in df.columns])

    print("\nK-Means Dataset Shape:")
    print(X.shape)

    # Identify Features
    numerical_features, categorical_features = identity_features(X)

    print("\nNumerical Features:")
    print(numerical_features)
    print("\nCategorical Features:")
    print(categorical_features)

    one_hot_features = [
        "Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"
    ]
    ordinal_features = ["CollegeTier", "CGPA_Tier"]

    # Handle Missing Values
    # (your functions expect train/test pairs — pass X as both since
    # there's no separate test split for clustering)
    X, _, _ = handle_missing_values(X, X, numerical_features)
    print("\nMissing Value Handling Completed.")

    # Standardization — keep the scaler for inverse-transforming centers later
    X, _, scaler = standardize_data(X, X, numerical_features)
    print("Standardization Completed.")

    # One-Hot Encoding
    X, _, _ = one_hot_encode_data(X, X, one_hot_features)
    print("One-Hot Encoding Completed.")

    # Ordinal Encoding
    X, _, _ = ordinal_encode_data(X, X, ordinal_features)
    print("Ordinal Encoding Completed.")

    # Elbow + Silhouette method to help pick K
    wcss, silhouette_scores, suggested_k = find_optimal_k(X)

    # Select K (use the suggested one, or override manually)
    k = suggested_k

    # Create and Train Model
    model = create_model(k)
    model = train_model(model, X)

    # Labels from training (no need to call predict again)
    labels = model.labels_

    # Evaluate Model
    evaluate_model(model, X, labels)

    # Add Cluster Labels back to the original dataframe
    df["Cluster"] = labels

    print("\nCluster Distribution:")
    print(df["Cluster"].value_counts().sort_index())

    # Cluster Centers — inverse-transformed to real units for numerical features
    centers_df = get_cluster_centers(model, scaler, numerical_features, X.columns)
    print("\nCluster Centers (numerical features in original scale):")
    print(centers_df)

    # Display Clusters via PCA
    display_clusters_pca(X, labels, model)

    # Save Results
    output_path = BASE_DIR / "data" / "kmeans_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nK-Means results saved successfully to {output_path}!")


if __name__ == "__main__":
    main()