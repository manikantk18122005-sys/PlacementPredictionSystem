import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(n_clusters=k,
                                    linkage='ward')
    return model

def train_model(model, X):
    model.fit(X)
    print("Agglomerative Clustering Result:")
    return model

def evaluate_model(model, X):
    labels = model.labels_
    score = silhouette_score(X, labels)
    print("Silhouette Result:")
    print(score)
    print(labels)

def display_dendogram(X, cut_distance):
    linked = linkage(X, method="ward")
    plt.figure(figsize=(12,6))
    dendrogram(linked,truncate_mode='lastp',p=30)
    plt.axhline(y=cut_distance,linestyle='--')
    plt.title('Dendogram Plot')


def main():
    df = load_data()
    print("Data Shape:")
    print(df.shape)

    df = df.sample(n=500,random_state=42)

    X = split_X_data(
        df, drop_columns=['StudentID', 'PlacementStatus', 'IsAnomaly', 'Salary Package']
    )

    numerical_features, categorical_features = identity_features(X)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']

    X, _, imputer = handle_missing_values(X, X, numerical_features)
    print("Missing values handled.")

    X, _, scaler = standardize_data(X, X, numerical_features)
    print("Standardization completed.")

    X, _, one_hot_encoder = one_hot_encode_data(X, X, one_hot_features)
    print("One-hot encoding completed.")

    X, _, ordinal_encoder = ordinal_encode_data(X, X, ordinal_features)
    print("Ordinal encoding completed.")

    cut_distance = 20

    linked = linkage(X, method="ward")
    clusters = fcluster(linked, t=cut_distance, criterion='distance')
    k = len(set(clusters))
    print("Clusters at cut_distance =", cut_distance, "->", k)

    model = create_model(k)
    model = train_model(model, X)
    evaluate_model(model, X)

    display_dendogram(X, cut_distance)
    plt.show()

if __name__ == "__main__":
    main()