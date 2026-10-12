"""Decision boundary visualization: t-SNE 2D projection of program consensus and disagreement."""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from solon.config.schema import DatasetConfig
from solon.datasets.preprocess import clean_and_impute
from solon.pipeline.program_synthesizer import load_program, run_program


def _get_decision_vector(code: str, cases_df: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    """Run a program on cases, return binary decision vector. 1=approve, 0=deny, -1=error."""
    approve_str = config.decisions.approve
    try:
        fn = load_program(code)
        predictions = run_program(fn, cases_df, config)
        return np.array([
            1 if p["decision"] == approve_str else (0 if p["error"] is None else -1)
            for p in predictions
        ], dtype=np.int8)
    except Exception:
        return np.full(len(cases_df), -1, dtype=np.int8)


def discover_program_files(dirs: list[str]) -> list[str]:
    """Find all valid .py candidate files in evolve output directories."""
    paths = []
    for d in dirs:
        py_files = sorted(glob.glob(os.path.join(d, "generations", "gen_*", "*.py")))
        paths.extend(p for p in py_files if "_eval" not in p)
    # Also check for top-level best_*.py
    for d in dirs:
        py_files = sorted(glob.glob(os.path.join(d, "best_*.py")))
        paths.extend(py_files)
    return paths


def compute_consensus_and_disagreement(
    program_paths: list[str],
    full_df: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run all programs on dataset, return (consensus, disagreement) arrays.

    consensus: approval rate per case [0, 1]
    disagreement: min(approval_rate, 1 - approval_rate) per case [0, 0.5]
    """
    from tqdm import tqdm

    codes = []
    for path in program_paths:
        with open(path) as f:
            codes.append(f.read())

    vectors = []
    for code in tqdm(codes, desc="Running programs"):
        vec = _get_decision_vector(code, full_df, config)
        vectors.append(vec)

    if not vectors:
        n = len(full_df)
        return np.full(n, 0.5), np.zeros(n)

    matrix = np.stack(vectors)
    valid_mask = matrix >= 0
    n_valid = valid_mask.sum(axis=0)

    with np.errstate(invalid="ignore"):
        consensus = np.where(
            n_valid > 0,
            (matrix * valid_mask).clip(0).sum(axis=0) / n_valid,
            0.5,
        )

    disagreement = np.minimum(consensus, 1 - consensus)
    disagreement = np.where(n_valid > 0, disagreement, 0.0)

    return consensus, disagreement


def tsne_reduce(full_df: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    """t-SNE-reduce dataset features to 2D. Returns (n_cases, 2) array."""
    from sklearn.manifold import TSNE

    clean = clean_and_impute(full_df, config)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    return reducer.fit_transform(X_scaled)


def plot_decision_landscape(
    coords: np.ndarray,
    consensus: np.ndarray,
    disagreement: np.ndarray,
    output_path: str,
    labeled_mask: np.ndarray = None,
    actual_outcomes: np.ndarray = None,
    title_prefix: str = "",
):
    """Generate 2D decision boundary scatter plots.

    Args:
        coords: (n_cases, 2) t-SNE coordinates.
        consensus: approval rate per case.
        disagreement: disagreement score per case.
        output_path: where to save the PNG.
        labeled_mask: bool array marking which cases are labeled.
        actual_outcomes: actual target column values (for ground truth panel).
        title_prefix: prepended to subplot titles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    x, y = coords[:, 0], coords[:, 1]
    n_panels = 3 if actual_outcomes is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 2:
        axes = list(axes)

    # Panel 1: consensus approval rate
    ax = axes[0]
    deny_approve_cmap = LinearSegmentedColormap.from_list(
        "deny_approve", ["#d62728", "#f0f0f0", "#2ca02c"]
    )
    sc = ax.scatter(x, y, c=consensus, cmap=deny_approve_cmap, s=3, alpha=0.6,
                    vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Consensus approval rate", shrink=0.8)
    if labeled_mask is not None:
        ax.scatter(x[labeled_mask], y[labeled_mask], facecolors="none",
                   edgecolors="black", s=20, linewidths=0.5, alpha=0.8, label="Labeled")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{title_prefix}Program Consensus", fontsize=13)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    # Panel 2: disagreement
    ax = axes[1]
    sc = ax.scatter(x, y, c=disagreement, cmap="inferno", s=3, alpha=0.6,
                    vmin=0, vmax=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Disagreement", shrink=0.8)
    if labeled_mask is not None:
        ax.scatter(x[labeled_mask], y[labeled_mask], facecolors="none",
                   edgecolors="cyan", s=20, linewidths=0.5, alpha=0.8, label="Labeled")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{title_prefix}Disagreement", fontsize=13)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    # Panel 3: actual outcomes (if available) — expects 1=approve, 0=deny
    if actual_outcomes is not None:
        ax = axes[2]
        sc = ax.scatter(x, y, c=actual_outcomes, cmap=deny_approve_cmap, s=3, alpha=0.6,
                        vmin=0, vmax=1, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Actual outcome (green=approve)", shrink=0.8)
        ax.set_title(f"{title_prefix}Actual Outcomes", fontsize=13)
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
