"""Ensemble majority voting across multiple model target files."""

import json
import pandas as pd
import numpy as np
from collections import Counter

from solon.config.schema import DatasetConfig


def ensemble_vote(
    all_targets: dict[str, list[dict]],
    cases: pd.DataFrame,
    config: DatasetConfig,
    output_path: str = None,
) -> list[dict]:
    """Perform majority voting across multiple sets of targets.

    Args:
        all_targets: dict mapping model name to list of target dicts.
        cases: DataFrame with actual outcomes.
        config: Dataset configuration.
        output_path: Optional path to save ensemble results.

    Returns list of ensemble result dicts.
    """
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny

    names = list(all_targets.keys())
    n_cases = len(cases)
    ensemble_results = []
    tp = fp = fn = tn = 0

    for i in range(n_cases):
        votes = {}
        reasons_all = []
        confidences = []

        for name in names:
            for t in all_targets[name]:
                if t["case_index"] == i:
                    d = t["decision"]
                    if d in (approve_str, deny_str):
                        votes[name] = d
                        reasons_all.extend(t.get("reason_codes", []))
                        conf = t.get("confidence", 0)
                        if isinstance(conf, (int, float)):
                            confidences.append(conf)
                    break

        vote_counts = Counter(votes.values())
        if not vote_counts:
            decision = "ERROR"
        else:
            decision = vote_counts.most_common(1)[0][0]

        actual = cases.iloc[i][target_col]
        if decision == approve_str and actual == pos_label:
            tp += 1
        elif decision == approve_str and actual == neg_label:
            fp += 1
        elif decision == deny_str and actual == pos_label:
            fn += 1
        elif decision == deny_str and actual == neg_label:
            tn += 1

        ensemble_results.append({
            "case_index": i,
            "decision": decision,
            "votes": votes,
            "unanimous": len(set(votes.values())) <= 1,
            "vote_counts": dict(vote_counts),
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "reason_codes": [r for r, _ in Counter(reasons_all).most_common(4)],
        })

    total = tp + fp + fn + tn
    print(f"{'='*60}")
    print(f"  ENSEMBLE RESULTS (majority vote, {len(names)} models)")
    print(f"{'='*60}")
    print(f"  Decisions:        {tp+fp} approve, {tn+fn} deny")
    if total:
        print(f"  Agreement:        {(tp+tn)/total:.1%} ({tp+tn}/{total})")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    print(f"  F1:               {f1:.3f}")
    print(f"  False approves:   {fp}")
    print(f"  False denies:     {fn}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(ensemble_results, f, indent=2)
        print(f"\n  Saved to {output_path}")

    return ensemble_results
