"""Select representative cases for LLM target generation.

Strategy: stratified sample covering the full risk spectrum + edge cases.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from solon.config.schema import DatasetConfig
from solon.datasets.preprocess import clean_and_impute, get_feature_columns, replace_special_values


def select_cases(
    df: pd.DataFrame,
    config: DatasetConfig,
    n_cases: int = None,
    n_clusters: int = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select representative cases using k-means clustering.

    Returns a DataFrame with n_cases rows sampled from n_clusters clusters,
    with a mix of outcomes in each cluster.
    """
    n_cases = n_cases or config.defaults.sample_size
    n_clusters = n_clusters or config.defaults.n_clusters
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label

    np.random.seed(random_state)

    # Clean and scale features
    clean = clean_and_impute(df, config)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    # Cluster
    per_cluster = max(1, n_cases // n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # From each cluster, pick a mix of outcomes
    selected = []
    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        good = cluster_df[cluster_df[target_col] == pos_label]
        bad = cluster_df[cluster_df[target_col] == neg_label]

        n_good = min(per_cluster // 2 + 1, len(good))
        n_bad = min(per_cluster - n_good, len(bad))
        n_good = min(per_cluster - n_bad, len(good))

        if len(good) > 0:
            selected.append(good.sample(n=n_good, random_state=random_state))
        if len(bad) > 0:
            selected.append(bad.sample(n=n_bad, random_state=random_state))

    sample_df = pd.concat(selected).head(n_cases)
    sample_df = sample_df.drop(columns=["cluster"])

    return sample_df


class ClusterMatcher:
    """Assigns test cases to their nearest cluster and provides matched few-shot examples."""

    def __init__(self, fewshot_csv: str, clustered_csv: str, config: DatasetConfig):
        from sklearn.preprocessing import StandardScaler

        fewshot_df = pd.read_csv(fewshot_csv)
        clustered_df = pd.read_csv(clustered_csv)

        target_col = config.target.column
        all_features = [c for c in clustered_df.columns if c not in (target_col, "cluster")]
        # Only use numeric features for clustering/scaling
        numeric_df = clustered_df[all_features].select_dtypes(include="number")
        features = list(numeric_df.columns)

        # Fit scaler on full clustered data (cleaned)
        clean = replace_special_values(numeric_df, config)
        self.medians = clean.median()
        self.scaler = StandardScaler()
        self.scaler.fit(clean.fillna(self.medians))

        # Compute cluster centroids in scaled space
        self.centroids = {}
        for c in clustered_df["cluster"].unique():
            mask = clustered_df["cluster"] == c
            cluster_clean = replace_special_values(
                numeric_df.loc[mask], config
            ).fillna(self.medians)
            self.centroids[c] = self.scaler.transform(cluster_clean).mean(axis=0)

        # Group few-shot examples by cluster
        self.examples_by_cluster = {}
        for _, row in fewshot_df.iterrows():
            c = int(row["cluster"])
            if c not in self.examples_by_cluster:
                self.examples_by_cluster[c] = []
            self.examples_by_cluster[c].append(row)

        self.features = features
        self.config = config

    def nearest_cluster(self, row: pd.Series) -> int:
        vals = replace_special_values(row[self.features], self.config).fillna(self.medians)
        scaled = self.scaler.transform(vals.values.reshape(1, -1))[0]
        best_c, best_d = None, float("inf")
        for c, centroid in self.centroids.items():
            d = np.linalg.norm(scaled - centroid)
            if d < best_d:
                best_c, best_d = c, d
        return best_c

    def format_matched_examples(self, cluster_id: int) -> str:
        examples = self.examples_by_cluster.get(cluster_id, [])
        if not examples:
            return ""

        target_col = self.config.target.column
        neg_label = self.config.target.negative_label
        special = set(self.config.special_values.values)
        meanings = self.config.special_values.meanings

        parts = ["  Similar reference cases with known outcomes:"]
        for row in examples:
            if target_col in row.index:
                outcome = self.config.target.negative_meaning if row[target_col] == neg_label else self.config.target.positive_meaning
            else:
                outcome = "unknown"
            parts.append(f"    [Outcome: {outcome.upper()}]")
            for col, val in row.items():
                if col in (target_col, "cluster"):
                    continue
                if val in special:
                    parts.append(f"      {col}: {meanings.get(int(val), 'special')}")
                else:
                    parts.append(f"      {col}: {val}")
        return "\n".join(parts)
