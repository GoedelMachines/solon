"""Compare target generation results across models against actual outcomes."""

import json
import pandas as pd
import numpy as np
from collections import Counter

from solon.config.schema import DatasetConfig


def score_targets(targets: list[dict], cases: pd.DataFrame, config: DatasetConfig) -> dict:
    """Score a set of targets against actual outcomes."""
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny

    tp = fp = fn = tn = 0
    false_approves = []
    false_denies = []
    confidences = []

    for t in targets:
        i = t["case_index"]
        actual = cases.iloc[i][target_col]
        decision = t["decision"]
        conf = t.get("confidence", 0)
        if isinstance(conf, (int, float)):
            confidences.append(conf)

        if decision == approve_str and actual == pos_label:
            tp += 1
        elif decision == approve_str and actual == neg_label:
            fp += 1
            false_approves.append(t)
        elif decision == deny_str and actual == pos_label:
            fn += 1
            false_denies.append(t)
        elif decision == deny_str and actual == neg_label:
            tn += 1

    total = tp + fp + fn + tn

    fd_reasons = []
    for t in false_denies:
        fd_reasons.extend(t.get("reason_codes", []))
    fa_reasons = []
    for t in false_approves:
        fa_reasons.extend(t.get("reason_codes", []))

    return {
        "total": total,
        "approve_count": tp + fp,
        "deny_count": tn + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "agreement": (tp + tn) / total if total else 0,
        "precision": tp / (tp + fp) if (tp + fp) else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "denial_precision": tn / (tn + fn) if (tn + fn) else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0,
        "avg_confidence": np.mean(confidences) if confidences else 0,
        "false_approve_count": fp,
        "false_deny_count": fn,
        "top_false_deny_reasons": Counter(fd_reasons).most_common(5),
        "top_false_approve_reasons": Counter(fa_reasons).most_common(5),
    }


def print_model_report(name: str, scores: dict):
    """Print a concise report for one model."""
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Decisions:        {scores['approve_count']} approve, {scores['deny_count']} deny")
    print(f"  Agreement:        {scores['agreement']:.1%} ({scores['tp'] + scores['tn']}/{scores['total']})")
    print(f"  Approval prec:    {scores['precision']:.1%}")
    print(f"  Approval recall:  {scores['recall']:.1%}")
    print(f"  Denial precision: {scores['denial_precision']:.1%}")
    print(f"  F1:               {scores['f1']:.3f}")
    print(f"  Avg confidence:   {scores['avg_confidence']:.2f}")
    print(f"  False approves:   {scores['false_approve_count']}")
    print(f"  False denies:     {scores['false_deny_count']}")

    if scores["top_false_deny_reasons"]:
        print(f"  Top false deny reasons:")
        for reason, count in scores["top_false_deny_reasons"]:
            print(f"    {count:>2}x  {reason}")


def print_comparison_table(all_scores: dict):
    """Print a side-by-side comparison table."""
    print(f"\n{'=' * 80}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 80}")

    names = list(all_scores.keys())
    col_width = max(18, max(len(n) for n in names) + 2)
    header = f"  {'Metric':<25}" + "".join(f"{n:>{col_width}}" for n in names)
    print(header)
    print(f"  {'─' * 25}" + "─" * col_width * len(names))

    metrics = [
        ("Approvals", "approve_count", "d"),
        ("Denials", "deny_count", "d"),
        ("Agreement", "agreement", ".1%"),
        ("Approval precision", "precision", ".1%"),
        ("Approval recall", "recall", ".1%"),
        ("Denial precision", "denial_precision", ".1%"),
        ("F1 score", "f1", ".3f"),
        ("Avg confidence", "avg_confidence", ".2f"),
        ("False approves", "false_approve_count", "d"),
        ("False denies", "false_deny_count", "d"),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:<25}"
        for name in names:
            val = all_scores[name][key]
            row += f"{format(val, fmt):>{col_width}}"
        print(row)
