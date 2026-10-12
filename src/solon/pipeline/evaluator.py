"""Multi-objective scoring of candidate programs against LLM-generated targets.

Hard constraints (reject) + soft objectives (weighted score).
"""

import re
import pandas as pd
import numpy as np
from solon.config.schema import DatasetConfig
from solon.pipeline.program_synthesizer import load_program, run_program


def count_conditions(code: str) -> int:
    """Count the number of conditional branches in the code."""
    return len(re.findall(r'\bif\b|\belif\b', code))


def evaluate_candidate(
    code: str,
    targets: list[dict],
    cases_df: pd.DataFrame,
    config: DatasetConfig,
    holdout_df: pd.DataFrame = None,
    weights: dict = None,
) -> dict:
    """Evaluate a single candidate program.

    Returns dict with scores, metrics, pass/fail, and overall score.
    """
    weights = weights or config.evaluation.weights

    result = {
        "valid": False,
        "hard_constraint_pass": True,
        "hard_constraint_failures": [],
        "metrics": {},
        "overall_score": 0.0,
    }

    # --- Load the program ---
    try:
        fn = load_program(code)
    except Exception as e:
        result["hard_constraint_failures"].append(f"Failed to load: {e}")
        result["hard_constraint_pass"] = False
        return result

    result["valid"] = True

    # --- Run on target cases ---
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny
    target_decisions = {
        t["case_index"]: t["decision"]
        for t in targets
        if t.get("decision") in (approve_str, deny_str)
    }
    predictions = run_program(fn, cases_df, config)

    # --- Hard Constraints ---

    # 1. No crashes: all predictions must succeed
    errors = [p for p in predictions if p["error"] is not None]
    error_rate = len(errors) / len(predictions) if predictions else 1.0
    if error_rate > config.evaluation.max_crash_rate:
        result["hard_constraint_failures"].append(
            f"Crash rate {error_rate:.1%} > {config.evaluation.max_crash_rate:.0%}"
        )
        result["hard_constraint_pass"] = False

    # 2. Every DENY must have at least one reason code
    if config.evaluation.require_reason_codes:
        denials_without_reasons = sum(
            1 for p in predictions
            if p["decision"] == deny_str and len(p.get("reason_codes", [])) == 0
        )
        if denials_without_reasons > 0:
            result["hard_constraint_failures"].append(
                f"{denials_without_reasons} {deny_str} decisions without reason codes"
            )
            result["hard_constraint_pass"] = False

    # --- Soft Objectives ---

    # 1. ACCURACY: Agreement with LLM-generated targets
    correct = 0
    total_compared = 0
    misclassifications = []
    summary_features = config.summary_features

    for i, pred in enumerate(predictions):
        if i in target_decisions and pred["error"] is None:
            if pred["decision"] == target_decisions[i]:
                correct += 1
            else:
                row = cases_df.iloc[i]
                feat_snapshot = {}
                for sf in summary_features:
                    val = row.get(sf, None)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        feat_snapshot[sf] = None
                    else:
                        feat_snapshot[sf] = val
                misclassifications.append({
                    "case_index": i,
                    "predicted": pred["decision"],
                    "expected": target_decisions[i],
                    "features": feat_snapshot,
                })
            total_compared += 1

    accuracy = correct / total_compared if total_compared > 0 else 0.0
    result["metrics"]["accuracy"] = accuracy
    result["metrics"]["correct"] = correct
    result["metrics"]["total_compared"] = total_compared
    result["misclassifications"] = misclassifications

    # 2. SIMPLICITY: Fewer conditions = more interpretable
    n_conditions = count_conditions(code)
    thresholds = config.evaluation.simplicity_thresholds
    ideal_max = thresholds.get("ideal_max", 10)
    good_max = thresholds.get("good_max", 20)
    acceptable_max = thresholds.get("acceptable_max", 40)

    if n_conditions <= ideal_max:
        simplicity = 1.0
    elif n_conditions <= good_max:
        simplicity = 1.0 - (n_conditions - ideal_max) / (good_max - ideal_max) * 0.5
    elif n_conditions <= acceptable_max:
        simplicity = 0.5 - (n_conditions - good_max) / (acceptable_max - good_max) * 0.25
    else:
        simplicity = max(0.0, 0.25 - (n_conditions - acceptable_max) / 100)
    result["metrics"]["simplicity"] = simplicity
    result["metrics"]["n_conditions"] = n_conditions
    result["metrics"]["code_lines"] = len(code.strip().split("\n"))

    # 3. ROBUSTNESS: Performance on holdout data (if available)
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label

    if holdout_df is not None:
        holdout_preds = run_program(fn, holdout_df, config)
        holdout_errors = sum(1 for p in holdout_preds if p["error"] is not None)
        crash_robustness = 1.0 - (holdout_errors / len(holdout_preds)) if holdout_preds else 0.0

        if target_col in holdout_df.columns:
            approvals_correct = 0
            approvals_total = 0
            denials_correct = 0
            denials_total = 0
            for j, pred in enumerate(holdout_preds):
                if pred["error"] is not None:
                    continue
                actual = holdout_df.iloc[j][target_col]
                if pred["decision"] == approve_str:
                    approvals_total += 1
                    if actual == pos_label:
                        approvals_correct += 1
                elif pred["decision"] == deny_str:
                    denials_total += 1
                    if actual == neg_label:
                        denials_correct += 1
            approval_precision = approvals_correct / approvals_total if approvals_total > 0 else 0.0
            denial_precision = denials_correct / denials_total if denials_total > 0 else 0.0
            outcome_quality = (approval_precision + denial_precision) / 2
            result["metrics"]["holdout_approval_precision"] = approval_precision
            result["metrics"]["holdout_denial_precision"] = denial_precision
            result["metrics"]["holdout_outcome_quality"] = outcome_quality
            robustness = 0.5 * crash_robustness + 0.5 * outcome_quality
        else:
            robustness = crash_robustness
    else:
        robustness = 1.0 - error_rate
    result["metrics"]["robustness"] = robustness

    # 4. COVERAGE: % of cases with non-empty reason codes
    cases_with_reasons = sum(
        1 for p in predictions
        if p["error"] is None and len(p.get("reason_codes", [])) > 0
    )
    valid_predictions = sum(1 for p in predictions if p["error"] is None)
    coverage = cases_with_reasons / valid_predictions if valid_predictions > 0 else 0.0
    result["metrics"]["coverage"] = coverage

    # --- Decision distribution ---
    approve_count = sum(1 for p in predictions if p["decision"] == approve_str)
    deny_count = sum(1 for p in predictions if p["decision"] == deny_str)
    error_count = sum(1 for p in predictions if p["decision"] == "ERROR")
    result["metrics"]["approve_count"] = approve_count
    result["metrics"]["deny_count"] = deny_count
    result["metrics"]["error_count"] = error_count

    # --- Overall Score ---
    if result["hard_constraint_pass"]:
        result["overall_score"] = (
            weights["accuracy"] * accuracy
            + weights["simplicity"] * simplicity
            + weights["robustness"] * robustness
            + weights["coverage"] * coverage
        )
    else:
        result["overall_score"] = 0.0

    return result


def rank_candidates(evaluations: list[dict]) -> list[int]:
    """Return indices of candidates sorted by overall score (best first)."""
    scores = [(i, e["overall_score"]) for i, e in enumerate(evaluations)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores]


def print_evaluation(eval_result: dict, candidate_id: int = 0):
    """Pretty-print an evaluation result."""
    m = eval_result["metrics"]
    print(f"\n  Candidate {candidate_id}:")
    print(f"    Valid: {eval_result['valid']}")
    print(f"    Hard constraints: {'PASS' if eval_result['hard_constraint_pass'] else 'FAIL'}")
    if eval_result["hard_constraint_failures"]:
        for f in eval_result["hard_constraint_failures"]:
            print(f"      FAIL: {f}")
    print(f"    Accuracy:   {m.get('accuracy', 0):.1%} ({m.get('correct', 0)}/{m.get('total_compared', 0)})")
    print(f"    Simplicity: {m.get('simplicity', 0):.2f} ({m.get('n_conditions', 0)} conditions, {m.get('code_lines', 0)} lines)")
    print(f"    Robustness: {m.get('robustness', 0):.2f}")
    print(f"    Coverage:   {m.get('coverage', 0):.1%}")
    print(f"    Decisions:  {m.get('approve_count', 0)} approve, {m.get('deny_count', 0)} deny, {m.get('error_count', 0)} error")
    print(f"    OVERALL:    {eval_result['overall_score']:.3f}")
