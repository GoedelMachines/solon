"""Evaluate a synthesized program directly against actual outcomes."""

import importlib.util
import pandas as pd
import numpy as np
from solon.config.schema import DatasetConfig


def load_underwriter(code_path: str):
    """Load the main function from a .py file."""
    spec = importlib.util.spec_from_file_location("_synth", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("underwrite", "evaluate", "decide"):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"No callable function found in {code_path}")


def evaluate(code_path: str, cases_path: str, config: DatasetConfig):
    """Evaluate a program against actual outcomes and print results."""
    cases = pd.read_csv(cases_path)
    underwrite = load_underwriter(code_path)
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny

    tp = fp = fn = tn = 0
    errors = 0
    decisions = []
    scores = []
    actuals = []

    for _, row in cases.iterrows():
        actual = row[target_col]
        app = row.drop(labels=[target_col]).to_dict()

        try:
            result = underwrite(app)
            decision = result.get("decision", "ERROR")
            score = result.get("score", None)
        except Exception:
            decision = "ERROR"
            score = None
            errors += 1

        decisions.append(decision)
        actuals.append(actual)
        scores.append(score)

        if decision == approve_str and actual == pos_label:
            tp += 1
        elif decision == approve_str and actual == neg_label:
            fp += 1
        elif decision == deny_str and actual == pos_label:
            fn += 1
        elif decision == deny_str and actual == neg_label:
            tn += 1

    total = tp + fp + fn + tn
    approve_count = tp + fp
    deny_count = tn + fn

    n_cases = len(cases)
    dataset_label = "FULL DATASET" if n_cases > 1000 else f"{n_cases}-CASE SAMPLE"

    print(f"\n{'='*60}")
    print(f"  {code_path}")
    print(f"  Evaluated against: ACTUAL OUTCOMES ({dataset_label}, n={n_cases})")
    print(f"{'='*60}")
    print(f"  Decisions:        {approve_count} approve, {deny_count} deny, {errors} errors")
    if total:
        print(f"  Agreement:        {(tp+tn)/total:.1%} ({tp+tn}/{total})")
    if tp + fp:
        print(f"  Approval prec:    {tp/(tp+fp):.1%}")
    if tp + fn:
        print(f"  Approval recall:  {tp/(tp+fn):.1%}")
    if tn + fn:
        print(f"  Denial precision: {tn/(tn+fn):.1%}")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    print(f"  F1:               {f1:.3f}")
    print(f"  False approves:   {fp}")
    print(f"  False denies:     {fn}")

    # AUC
    from sklearn.metrics import roc_auc_score
    valid_scores = [(s, a) for s, a in zip(scores, actuals) if s is not None]
    if valid_scores:
        s_arr = np.array([x[0] for x in valid_scores])
        a_arr = np.array([x[1] for x in valid_scores])
        try:
            auc = roc_auc_score(a_arr, s_arr)
            print(f"  AUC (score):      {auc:.3f}")
        except ValueError:
            print(f"  AUC (score):      N/A (single class)")

    binary_preds = [1 if d == deny_str else 0 for d, a in zip(decisions, actuals) if d in (approve_str, deny_str)]
    binary_acts = [a for d, a in zip(decisions, actuals) if d in (approve_str, deny_str)]
    if binary_preds:
        try:
            auc_bin = roc_auc_score(binary_acts, binary_preds)
            print(f"  AUC (decision):   {auc_bin:.3f}")
        except ValueError:
            pass
