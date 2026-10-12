"""K-Means clustering of the dataset to identify natural archetypes.

Outputs cluster profiles and selects representative cases from each cluster.
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from solon.config.schema import DatasetConfig
from solon.datasets.preprocess import clean_and_impute, get_feature_columns


def cluster_dataset(
    df: pd.DataFrame,
    config: DatasetConfig,
    output_dir: str = "outputs",
    min_k: int = 4,
    max_k: int = 15,
):
    """Cluster the dataset and select representative cases.

    Returns (clustered_df, fewshot_df, cluster_profiles, selected_cases).
    """
    target_col = config.target.column
    features = get_feature_columns(df, config)
    clean = clean_and_impute(df, config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    # Find optimal K
    print("Finding optimal number of clusters...")
    silhouette_scores = {}
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        silhouette_scores[k] = sil
        print(f"  k={k}: silhouette={sil:.3f}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\nBest k={best_k} (silhouette={silhouette_scores[best_k]:.3f})")

    # Cluster with best K
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Profile clusters
    cluster_profiles = []
    for c in range(best_k):
        mask = df["cluster"] == c
        cluster_df = df[mask]
        n = len(cluster_df)
        bad_rate = cluster_df[target_col].mean() if target_col in cluster_df.columns else 0

        profile = {"cluster": c, "n": n, "pct": n / len(df) * 100, "bad_rate": bad_rate}
        for feat in config.summary_features:
            if feat in cluster_df.columns:
                profile[f"{feat}_median"] = cluster_df[feat].median()
        cluster_profiles.append(profile)

    # Select representative cases
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X_scaled)

    selected_cases = []
    for c in range(best_k):
        mask = df["cluster"] == c
        cluster_indices = df[mask].index.tolist()
        centroid = kmeans.cluster_centers_[c]

        # Closest to centroid
        distances = np.linalg.norm(X_scaled[cluster_indices] - centroid, axis=1)
        centroid_case = cluster_indices[np.argmin(distances)]

        # Boundary case
        best_boundary_case = None
        best_boundary_mix = 1.0
        for idx in cluster_indices:
            dists, neighbors = nn.kneighbors(X_scaled[idx].reshape(1, -1))
            if target_col in df.columns:
                neighbor_outcomes = df.iloc[neighbors[0]][target_col].values
                mix = abs(neighbor_outcomes.mean() - 0.5)
            else:
                mix = 0.5
            if mix < best_boundary_mix:
                best_boundary_mix = mix
                best_boundary_case = idx

        selected_cases.append({"cluster": c, "type": "centroid", "index": int(centroid_case)})
        if best_boundary_case is not None:
            selected_cases.append({"cluster": c, "type": "boundary", "index": int(best_boundary_case)})

    selected_indices = list(dict.fromkeys([s["index"] for s in selected_cases]))
    fewshot_df = df.loc[selected_indices].copy()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "clustered.csv"), index=False)
    fewshot_df.to_csv(os.path.join(output_dir, "fewshot_cases.csv"), index=False)
    with open(os.path.join(output_dir, "cluster_profiles.json"), "w") as f:
        json.dump(cluster_profiles, f, indent=2)
    with open(os.path.join(output_dir, "selected_cases.json"), "w") as f:
        json.dump(selected_cases, f, indent=2)

    print(f"\nSaved clustering outputs to {output_dir}/")
    return df, fewshot_df, cluster_profiles, selected_cases
