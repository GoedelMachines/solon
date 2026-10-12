"""UMAP + HDBSCAN clustering for density-based archetype discovery.

Better cluster separation than K-Means on high-dimensional credit data.
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from solon.config.schema import DatasetConfig
from solon.datasets.preprocess import clean_and_impute, get_feature_columns


def cluster_dataset_umap(
    df: pd.DataFrame,
    config: DatasetConfig,
    output_dir: str = "outputs",
    n_umap_components: int = 10,
    min_cluster_size: int = 100,
    min_samples: int = 20,
):
    """Cluster using UMAP dimensionality reduction + HDBSCAN.

    Returns (clustered_df, fewshot_df, cluster_profiles, selected_cases).
    """
    import umap
    import hdbscan

    target_col = config.target.column
    features = get_feature_columns(df, config)
    clean = clean_and_impute(df, config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    # UMAP reduction
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_components=n_umap_components,
        n_neighbors=30,
        min_dist=0.0,
        metric="euclidean",
        random_state=42,
    )
    X_umap = reducer.fit_transform(X_scaled)
    print(f"  Reduced {X_scaled.shape[1]} features -> {X_umap.shape[1]} UMAP components")

    # HDBSCAN clustering
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_umap)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(df)*100:.1f}%)")

    # Assign noise points to nearest cluster
    if n_noise > 0:
        from sklearn.neighbors import KNeighborsClassifier
        mask_clustered = labels != -1
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_umap[mask_clustered], labels[mask_clustered])
        labels[~mask_clustered] = knn.predict(X_umap[~mask_clustered])

    df = df.copy()
    df["cluster"] = labels

    # Profile clusters
    cluster_profiles = []
    for c in sorted(df["cluster"].unique()):
        mask = df["cluster"] == c
        cluster_df = df[mask]
        n = len(cluster_df)
        bad_rate = cluster_df[target_col].mean() if target_col in cluster_df.columns else 0

        profile = {"cluster": int(c), "n": n, "pct": n / len(df) * 100, "bad_rate": bad_rate}
        for feat in config.summary_features:
            if feat in cluster_df.columns:
                profile[f"{feat}_median"] = cluster_df[feat].median()
        cluster_profiles.append(profile)

    # Select representative cases
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X_umap)

    selected_cases = []
    for c in sorted(df["cluster"].unique()):
        mask = df["cluster"] == c
        cluster_indices = df[mask].index.tolist()
        cluster_umap = X_umap[cluster_indices]
        centroid = cluster_umap.mean(axis=0)

        distances = np.linalg.norm(cluster_umap - centroid, axis=1)
        centroid_case = cluster_indices[np.argmin(distances)]

        best_boundary_case = None
        best_boundary_mix = 1.0
        for idx in cluster_indices:
            dists, neighbors = nn.kneighbors(X_umap[idx].reshape(1, -1))
            if target_col in df.columns:
                neighbor_outcomes = df.iloc[neighbors[0]][target_col].values
                mix = abs(neighbor_outcomes.mean() - 0.5)
            else:
                mix = 0.5
            if mix < best_boundary_mix:
                best_boundary_mix = mix
                best_boundary_case = idx

        selected_cases.append({"cluster": int(c), "type": "centroid", "index": int(centroid_case)})
        if best_boundary_case is not None:
            selected_cases.append({"cluster": int(c), "type": "boundary", "index": int(best_boundary_case)})

    selected_indices = list(dict.fromkeys([s["index"] for s in selected_cases]))
    fewshot_df = df.loc[selected_indices].copy()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "clustered_umap.csv"), index=False)
    fewshot_df.to_csv(os.path.join(output_dir, "fewshot_cases_umap.csv"), index=False)
    with open(os.path.join(output_dir, "cluster_profiles_umap.json"), "w") as f:
        json.dump(cluster_profiles, f, indent=2)
    with open(os.path.join(output_dir, "selected_cases_umap.json"), "w") as f:
        json.dump(selected_cases, f, indent=2)

    # 2D t-SNE for visualization
    print("Generating 2D t-SNE embedding for visualization...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_scaled)
    viz_df = pd.DataFrame({
        "tsne_x": X_2d[:, 0], "tsne_y": X_2d[:, 1],
        "cluster": labels,
    })
    if target_col in df.columns:
        viz_df["target"] = df[target_col].values
    viz_df.to_csv(os.path.join(output_dir, "tsne_2d_viz.csv"), index=False)

    print(f"\nSaved clustering outputs to {output_dir}/")
    return df, fewshot_df, cluster_profiles, selected_cases
