"""Program similarity analysis: behavioral (decision vectors) and structural (code tokens).

Compares all candidate programs by how they behave on a dataset and
by code-level token similarity. Outputs pairwise matrices, t-SNE 2D
projections, and hierarchical cluster assignments.
"""

import glob
import importlib.util
import os
import re
import tokenize
import io

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.preprocessing import StandardScaler

from solon.config.schema import DatasetConfig
from solon.datasets.preprocess import clean_and_impute, get_feature_columns


def discover_programs(dirs: list[str]) -> list[dict]:
    """Find all .py candidates in evolve output directories."""
    programs = []
    for d in dirs:
        py_files = sorted(glob.glob(os.path.join(d, "generations", "gen_*", "*.py")))
        py_files = [p for p in py_files if "_eval" not in p]
        run_name = os.path.basename(d.rstrip("/"))
        for p in py_files:
            gen = p.split("/gen_")[1].split("/")[0] if "/gen_" in p else "?"
            cand = os.path.basename(p).replace(".py", "")
            programs.append({
                "run": run_name,
                "gen": gen,
                "candidate": cand,
                "path": p,
                "label": f"{run_name}/g{gen}/{cand}",
            })
    return programs


def _load_fn(path: str):
    """Load underwrite function from a .py file."""
    spec = importlib.util.spec_from_file_location("_synth", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("underwrite", "evaluate", "decide"):
        if hasattr(mod, name):
            return getattr(mod, name)
    return None


def build_decision_vectors(
    programs: list[dict],
    cases_df: pd.DataFrame,
    config: DatasetConfig,
) -> np.ndarray:
    """Run each program on all cases. Returns (n_programs, n_cases) binary matrix.

    1 = approve, 0 = deny, -1 = error.
    """
    target_col = config.target.column
    approve_str = config.decisions.approve
    apps = cases_df.drop(columns=[target_col], errors="ignore").to_dict("records")

    n_programs = len(programs)
    n_cases = len(apps)
    matrix = np.full((n_programs, n_cases), -1, dtype=np.int8)

    for i, prog in enumerate(programs):
        fn = _load_fn(prog["path"])
        if fn is None:
            continue
        for j, app in enumerate(apps):
            try:
                result = fn(app)
                decision = result.get("decision", "ERROR")
                matrix[i, j] = 1 if decision == approve_str else 0
            except Exception:
                matrix[i, j] = -1

    return matrix


def pairwise_agreement(decision_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise agreement rate between all program pairs.

    Ignores cases where either program errored (-1).
    Returns (n_programs, n_programs) matrix with values in [0, 1].
    """
    n = decision_matrix.shape[0]
    agreement = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            valid = (decision_matrix[i] >= 0) & (decision_matrix[j] >= 0)
            if valid.sum() == 0:
                agreement[i, j] = agreement[j, i] = 0.0
            else:
                agree = (decision_matrix[i, valid] == decision_matrix[j, valid]).mean()
                agreement[i, j] = agreement[j, i] = agree

    return agreement


def tokenize_code(source: str) -> str:
    """Tokenize Python source into space-separated tokens for TF-IDF.

    Strips comments and string literals, keeps names/keywords/operators.
    """
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.ENCODING, tokenize.NEWLINE,
                            tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
                            tokenize.ENDMARKER):
                continue
            if tok.type == tokenize.STRING:
                tokens.append("STR")
            elif tok.type == tokenize.NUMBER:
                tokens.append("NUM")
            else:
                tokens.append(tok.string)
    except tokenize.TokenError:
        # Fallback: split on whitespace
        tokens = source.split()
    return " ".join(tokens)


def code_similarity_matrix(programs: list[dict]) -> np.ndarray:
    """Compute pairwise code similarity using TF-IDF on Python tokens."""
    docs = []
    for prog in programs:
        with open(prog["path"]) as f:
            source = f.read()
        docs.append(tokenize_code(source))

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform(docs)
    return cosine_similarity(tfidf)


def cluster_programs(similarity_matrix: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Hierarchical clustering from a similarity matrix.

    Returns cluster labels (1-indexed).
    """
    distance = 1.0 - similarity_matrix
    np.fill_diagonal(distance, 0)
    distance = np.clip(distance, 0, 1)
    condensed = squareform(distance)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels


def tsne_2d(decision_matrix: np.ndarray) -> np.ndarray:
    """t-SNE 2D projection of decision vectors. Returns (n, 2) array."""
    from sklearn.manifold import TSNE

    clean = decision_matrix.copy().astype(np.float32)
    clean[clean < 0] = 0
    perplexity = min(30, max(1, len(clean) - 1))  # t-SNE needs perplexity < n_samples
    reducer = TSNE(n_components=2, perplexity=perplexity, metric="hamming", random_state=42)
    return reducer.fit_transform(clean)


def run_similarity_analysis(
    dirs: list[str],
    cases_df: pd.DataFrame,
    config: DatasetConfig,
    output_dir: str = "outputs/similarity",
) -> pd.DataFrame:
    """Full similarity pipeline. Returns summary DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    # Discover
    programs = discover_programs(dirs)
    if not programs:
        print("No programs found.")
        return pd.DataFrame()

    n = len(programs)
    print(f"Found {n} candidate programs across {len(dirs)} directories")

    # Decision vectors
    print(f"Building decision vectors ({n} programs x {len(cases_df)} cases)...")
    decisions = build_decision_vectors(programs, cases_df, config)

    # Behavioral similarity
    print("Computing behavioral similarity (pairwise agreement)...")
    behav_sim = pairwise_agreement(decisions)

    # Code similarity
    print("Computing code similarity (TF-IDF tokens)...")
    code_sim = code_similarity_matrix(programs)

    # Clustering (on behavioral similarity)
    print("Clustering programs by behavior...")
    behav_clusters = cluster_programs(behav_sim)
    code_clusters = cluster_programs(code_sim)

    # t-SNE projection
    print("Computing t-SNE 2D projection...")
    coords = tsne_2d(decisions)

    # Approve rates from decision vectors
    approve_rates = []
    for i in range(n):
        valid = decisions[i] >= 0
        if valid.sum() > 0:
            approve_rates.append(decisions[i, valid].mean())
        else:
            approve_rates.append(0.0)

    # Build summary DataFrame
    summary = pd.DataFrame({
        "label": [p["label"] for p in programs],
        "run": [p["run"] for p in programs],
        "gen": [p["gen"] for p in programs],
        "candidate": [p["candidate"] for p in programs],
        "path": [p["path"] for p in programs],
        "behav_cluster": behav_clusters,
        "code_cluster": code_clusters,
        "approve_rate": approve_rates,
        "tsne_x": coords[:, 0],
        "tsne_y": coords[:, 1],
    })

    # Save outputs
    summary.to_csv(os.path.join(output_dir, "program_summary.csv"), index=False)

    labels = [p["label"] for p in programs]
    behav_df = pd.DataFrame(behav_sim, index=labels, columns=labels)
    behav_df.to_csv(os.path.join(output_dir, "behavioral_similarity.csv"))

    code_df = pd.DataFrame(code_sim, index=labels, columns=labels)
    code_df.to_csv(os.path.join(output_dir, "code_similarity.csv"))

    np.save(os.path.join(output_dir, "decision_vectors.npy"), decisions)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Program Similarity Analysis: {n} programs")
    print(f"{'='*70}")

    # Behavioral clusters
    print(f"\nBehavioral clusters ({len(set(behav_clusters))}):")
    for c in sorted(set(behav_clusters)):
        members = summary[summary["behav_cluster"] == c]
        runs = members["run"].value_counts().to_dict()
        avg_approve = members["approve_rate"].mean()
        run_str = ", ".join(f"{r}:{cnt}" for r, cnt in sorted(runs.items()))
        print(f"  Cluster {c}: {len(members)} programs, approve={avg_approve:.1%}, models=[{run_str}]")

    # Code clusters
    print(f"\nCode clusters ({len(set(code_clusters))}):")
    for c in sorted(set(code_clusters)):
        members = summary[summary["code_cluster"] == c]
        runs = members["run"].value_counts().to_dict()
        run_str = ", ".join(f"{r}:{cnt}" for r, cnt in sorted(runs.items()))
        print(f"  Cluster {c}: {len(members)} programs, models=[{run_str}]")

    # Cross-model agreement
    print(f"\nAverage between-model behavioral agreement:")
    runs = summary["run"].unique()
    for i, r1 in enumerate(runs):
        for r2 in runs[i+1:]:
            idx1 = summary[summary["run"] == r1].index.tolist()
            idx2 = summary[summary["run"] == r2].index.tolist()
            agreements = []
            for a in idx1:
                for b in idx2:
                    agreements.append(behav_sim[a, b])
            if agreements:
                print(f"  {r1} vs {r2}: {np.mean(agreements):.1%}")

    # Most unique programs (lowest avg agreement with others)
    avg_agreement = behav_sim.mean(axis=1)
    most_unique_idx = np.argsort(avg_agreement)[:5]
    print(f"\nMost behaviorally unique programs:")
    for idx in most_unique_idx:
        print(f"  {labels[idx]}: avg agreement={avg_agreement[idx]:.1%}, approve={approve_rates[idx]:.1%}")

    # Most similar pairs
    print(f"\nMost behaviorally similar pairs (non-identical):")
    tri = np.triu(behav_sim, k=1)
    flat_indices = np.argsort(tri.ravel())[::-1]
    shown = 0
    for flat_idx in flat_indices:
        if shown >= 5:
            break
        i, j = np.unravel_index(flat_idx, tri.shape)
        val = tri[i, j]
        if val <= 0 or val >= 1.0:
            continue
        if programs[i]["run"] == programs[j]["run"] and programs[i]["gen"] == programs[j]["gen"]:
            continue  # skip same-run same-gen pairs (boring)
        print(f"  {labels[i]} <-> {labels[j]}: {val:.1%}")
        shown += 1

    # Decision landscape (phase diagram)
    print("\nGenerating decision landscape (t-SNE of data + program decisions)...")
    landscape_df = build_decision_landscape(programs, cases_df, config, decisions, output_dir)

    print(f"\nOutputs saved to {output_dir}/")
    return summary


def build_decision_landscape(
    programs: list[dict],
    cases_df: pd.DataFrame,
    config: DatasetConfig,
    decision_matrix: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """t-SNE-reduce the dataset features, then overlay program decisions.

    Produces a CSV with tsne_x, tsne_y, actual_outcome, and each program's
    decision for every case. This is the "phase diagram" showing which
    regions of feature space each program accepts/rejects.
    """
    from sklearn.manifold import TSNE

    target_col = config.target.column
    features = get_feature_columns(cases_df, config)
    clean = clean_and_impute(cases_df, config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = reducer.fit_transform(X_scaled)

    landscape = pd.DataFrame({
        "tsne_x": coords[:, 0],
        "tsne_y": coords[:, 1],
    })

    if target_col in cases_df.columns:
        landscape["actual"] = cases_df[target_col].values

    # Add each program's decisions (build as dict to avoid fragmentation)
    prog_cols = {}
    for i, prog in enumerate(programs):
        col_name = prog["label"].replace("/", "_")
        prog_cols[col_name] = decision_matrix[i]
    landscape = pd.concat([landscape, pd.DataFrame(prog_cols)], axis=1)

    # Consensus columns: what fraction of programs approve each case
    valid_mask = decision_matrix >= 0
    with np.errstate(invalid="ignore"):
        consensus = np.where(
            valid_mask.sum(axis=0) > 0,
            (decision_matrix * valid_mask).sum(axis=0) / valid_mask.sum(axis=0),
            0.0,
        )
    landscape["consensus_approve"] = consensus

    # Disagreement: std of decisions across programs (high = controversial case)
    landscape["disagreement"] = np.std(decision_matrix.clip(0, 1).astype(float), axis=0)

    landscape.to_csv(os.path.join(output_dir, "decision_landscape.csv"), index=False)

    # Print summary of controversial regions
    high_disagree = landscape[landscape["disagreement"] > 0.3]
    print(f"  Cases with high disagreement (>0.3 std): {len(high_disagree)}/{len(landscape)}")
    if target_col in cases_df.columns:
        if len(high_disagree) > 0:
            bad_rate = high_disagree["actual"].mean()
            overall_bad = landscape["actual"].mean()
            print(f"  Bad rate in high-disagreement region: {bad_rate:.1%} (vs {overall_bad:.1%} overall)")

    # Per-cluster consensus breakdown (use consensus quartiles as proxy clusters)
    landscape["consensus_bin"] = pd.cut(
        landscape["consensus_approve"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["mostly_deny", "lean_deny", "lean_approve", "mostly_approve"],
        include_lowest=True,
    )
    print("\n  Decision landscape by consensus:")
    for bin_name in ["mostly_deny", "lean_deny", "lean_approve", "mostly_approve"]:
        subset = landscape[landscape["consensus_bin"] == bin_name]
        if len(subset) == 0:
            continue
        if target_col in cases_df.columns:
            bad_rate = subset["actual"].mean()
            print(f"    {bin_name:>15}: {len(subset):>5} cases, bad_rate={bad_rate:.1%}")
        else:
            print(f"    {bin_name:>15}: {len(subset):>5} cases")

    return landscape
