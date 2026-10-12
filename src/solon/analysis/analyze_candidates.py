"""Analyze all synthesized underwriting candidates.

Code structure analysis and evaluation against actual outcomes.
"""

import re
import importlib.util
import pandas as pd
import numpy as np

from solon.config.schema import DatasetConfig


def evaluate_candidate_against_actuals(code_path: str, cases_df: pd.DataFrame, config: DatasetConfig) -> dict:
    """Evaluate a single candidate against actual outcomes. Returns metrics dict."""
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny

    apps = cases_df.drop(columns=[target_col], errors="ignore").to_dict("records")
    actuals = cases_df[target_col].values if target_col in cases_df.columns else []

    try:
        spec = importlib.util.spec_from_file_location("_synth", code_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = None
        for name in ("underwrite", "evaluate", "decide"):
            if hasattr(mod, name):
                fn = getattr(mod, name)
                break
        if fn is None:
            return {"accuracy": 0, "f1": 0, "approve_rate": 0, "error": "No function found"}
    except Exception as e:
        return {"accuracy": 0, "f1": 0, "approve_rate": 0, "error": str(e)}

    tp = fp = fn_count = tn = errors = 0
    for app, actual in zip(apps, actuals):
        try:
            result = fn(app)
            decision = result.get("decision", "ERROR")
        except Exception:
            decision = "ERROR"
            errors += 1

        if decision == approve_str and actual == pos_label:
            tp += 1
        elif decision == approve_str and actual == neg_label:
            fp += 1
        elif decision == deny_str and actual == pos_label:
            fn_count += 1
        elif decision == deny_str and actual == neg_label:
            tn += 1

    total = tp + fp + fn_count + tn
    accuracy = (tp + tn) / total if total else 0
    f1 = 2 * tp / (2 * tp + fp + fn_count) if (2 * tp + fp + fn_count) else 0
    approve_rate = (tp + fp) / total if total else 0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "approve_rate": approve_rate,
        "tp": tp, "fp": fp, "fn": fn_count, "tn": tn,
        "errors": errors,
    }


def analyze_code_structure(code: str) -> dict:
    """Extract structural features from source code."""
    features_used = set()
    for m in re.finditer(r'(?:application\.get|get_val|get_feat|get)\s*\(\s*["\'](\w+)["\']', code):
        features_used.add(m.group(1))

    conditions = len(re.findall(r'\bif\b|\belif\b', code))
    has_override = bool(re.search(r'override|compensat|exception|recover', code, re.I))

    return {
        "n_features": len(features_used),
        "features": sorted(features_used),
        "n_conditions": conditions,
        "has_override": has_override,
        "lines": len(code.strip().splitlines()),
    }
