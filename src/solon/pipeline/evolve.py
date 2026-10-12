"""Main pipeline: Oracle-Guided Evolutionary Program Synthesis.

The full loop:
1. Generate targets (LLM reasons about cases)
2. Synthesize candidate programs (LLM writes code)
3. Evaluate candidates against targets
4. Keep top-K, use as best-shot examples for next generation
5. Repeat
"""

import copy
import glob
import json
import os
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from solon.config.schema import RunConfig, DatasetConfig
from solon.pipeline.target_generator import generate_targets
from solon.pipeline.program_synthesizer import (
    synthesize_programs, validate_program, load_program, run_program,
)
from solon.pipeline.evaluator import evaluate_candidate, rank_candidates, print_evaluation


def _attach_case_features(targets: list[dict], cases_df: pd.DataFrame) -> list[dict]:
    """Fill missing `original_features` on targets from the cases CSV by `case_index`.

    Shipped example targets omit the raw dataset rows; the features are
    re-joined here from the user's own cases CSV.
    """
    for t in targets:
        i = t.get("case_index")
        if "original_features" not in t and i is not None and 0 <= i < len(cases_df):
            t["original_features"] = cases_df.iloc[i].to_dict()
    return targets


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


def _pairwise_disagreement(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Fraction of cases where two programs disagree (ignoring errors)."""
    valid = (vec_a >= 0) & (vec_b >= 0)
    if valid.sum() == 0:
        return 1.0
    return (vec_a[valid] != vec_b[valid]).mean()


def select_diverse_programs(
    candidates: list[dict],
    top_k: int,
    cases_df: pd.DataFrame,
    config: DatasetConfig,
) -> list[dict]:
    """Clade-based diversity selection.

    Instead of pure top-K by score, selects programs that maintain
    behavioral diversity across clades (distinct decision strategies).

    Strategy:
      1. Always keep the global best (elite preservation)
      2. For each remaining slot, pick the candidate that maximizes
         a quality + diversity score — must be within 85% of the best
         score, then prefer candidates most different from the
         already-selected set (maxmin disagreement).
    """
    if len(candidates) <= top_k:
        return candidates

    # Attach decision vectors
    for c in candidates:
        if "decisions" not in c:
            c["decisions"] = _get_decision_vector(c["code"], cases_df, config)

    best_score = max(c["score"] for c in candidates)
    quality_floor = best_score * 0.85

    selected = []
    selected_indices = set()
    idx_by_score = sorted(range(len(candidates)), key=lambda i: candidates[i]["score"], reverse=True)

    # 1. Elite: global best
    elite_idx = idx_by_score[0]
    selected.append(candidates[elite_idx])
    selected_indices.add(elite_idx)

    # 2. Fill remaining via quality-gated maxmin diversity
    while len(selected) < top_k:
        best_combined = -1
        best_pick = None

        for i in range(len(candidates)):
            if i in selected_indices:
                continue
            c = candidates[i]

            # Quality gate
            if c["score"] < quality_floor:
                continue

            # Min disagreement with any already-selected program
            min_disagree = 1.0
            for s in selected:
                d = _pairwise_disagreement(c["decisions"], s["decisions"])
                min_disagree = min(min_disagree, d)

            # Combined: normalize quality to [0,1] + diversity bonus
            norm_score = c["score"] / best_score if best_score > 0 else 0
            combined = 0.3 * norm_score + 0.7 * min_disagree

            if combined > best_combined:
                best_combined = combined
                best_pick = i

        if best_pick is None:
            # Relax quality floor: take next best by score
            for i in idx_by_score:
                if i not in selected_indices:
                    best_pick = i
                    break
            if best_pick is None:
                break

        selected.append(candidates[best_pick])
        selected_indices.add(best_pick)

    # Report diversity
    if len(selected) > 1:
        pairs = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pairs.append(_pairwise_disagreement(
                    selected[i]["decisions"], selected[j]["decisions"]
                ))
        avg_disagree = np.mean(pairs)
        models = set(s.get("source_model", "?") for s in selected)
        print(f"  Clade selection: {len(selected)} programs, "
              f"avg pairwise disagreement={avg_disagree:.1%}, "
              f"models={models}")
        for i, s in enumerate(selected):
            print(f"    Slot {i}: score={s['score']:.3f} "
                  f"[{s.get('source_model', '?')}] "
                  f"approve_rate={s['decisions'][s['decisions']>=0].mean():.1%}")

    return selected


def evolve(run_config: RunConfig, targets_path: str = None):
    """Run the full evolutionary pipeline.

    Args:
        run_config: Full run configuration.
        targets_path: Pre-generated targets JSON (skip target generation).
    """
    config = run_config.dataset
    output_dir = run_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations"), exist_ok=True)

    cases_df = pd.read_csv(run_config.cases_csv)

    # Optional holdout from full dataset
    holdout_df = None
    if run_config.full_dataset_csv:
        full_df = pd.read_csv(run_config.full_dataset_csv)
        holdout_df = full_df.sample(
            n=min(run_config.holdout_size, len(full_df)),
            random_state=42,
        )

    # Use provided targets_path or fall back to run_config
    targets_path = targets_path or run_config.targets_path

    # =========================================
    # STEP 1: Generate Targets (or load existing)
    # =========================================
    if targets_path and os.path.exists(targets_path):
        print("=" * 60)
        print("STEP 1: Loading pre-generated targets")
        print("=" * 60)
        with open(targets_path) as f:
            targets = _attach_case_features(json.load(f), cases_df)
        print(f"  Loaded {len(targets)} targets")
    else:
        print("=" * 60)
        print("STEP 1: Generating Targets (LLM reasons about each case)")
        print("=" * 60)
        targets_path = os.path.join(output_dir, "targets.json")
        targets = generate_targets(run_config, output_path=targets_path)

    # Summary
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny
    approve = sum(1 for t in targets if t.get("decision") == approve_str)
    deny = sum(1 for t in targets if t.get("decision") == deny_str)
    print(f"\n  Targets: {approve} {approve_str}, {deny} {deny_str}")
    print(f"  >>> REVIEW THESE TARGETS BEFORE PROCEEDING <<<")
    print(f"  File: {targets_path}")

    # =========================================
    # STEP 2-5: Evolutionary Loop
    # =========================================
    print("\n" + "=" * 60)
    print("STEPS 2-5: Evolutionary Program Synthesis")
    print("=" * 60)

    best_programs = []
    all_history = []

    for gen in range(run_config.n_generations):
        print(f"\n--- Generation {gen + 1}/{run_config.n_generations} ---")

        previous_best = best_programs[:run_config.top_k] if best_programs else None

        candidates = synthesize_programs(
            targets=targets,
            config=config,
            n_candidates=run_config.n_candidates,
            previous_best=previous_best,
            provider=run_config.provider,
            model=run_config.model,
        )

        gen_results = []
        for c in candidates:
            if not c["valid"]:
                gen_results.append({"candidate": c, "evaluation": {"overall_score": 0.0, "valid": False}})
                continue

            eval_result = evaluate_candidate(
                code=c["code"],
                targets=targets,
                cases_df=cases_df,
                config=config,
                holdout_df=holdout_df,
            )
            gen_results.append({"candidate": c, "evaluation": eval_result})
            print_evaluation(eval_result, c["candidate_id"])

        # Rank and keep top-K with full eval data
        valid_results = [
            {
                "code": c["candidate"]["code"],
                "score": c["evaluation"]["overall_score"],
                "metrics": c["evaluation"].get("metrics", {}),
                "misclassifications": c["evaluation"].get("misclassifications", []),
            }
            for c in gen_results if c["evaluation"].get("valid")
        ]

        all_programs = best_programs + valid_results
        all_programs.sort(key=lambda x: x["score"], reverse=True)
        best_programs = all_programs[:run_config.top_k]

        top_scores = [f"{p['score']:.3f}" for p in valid_results[:3]]
        print(f"\n  Top scores this generation: {top_scores}")
        print(f"  All-time best: {best_programs[0]['score']:.3f}" if best_programs else "  No valid programs yet")

        # Save generation state
        gen_dir = os.path.join(output_dir, "generations", f"gen_{gen+1:03d}")
        os.makedirs(gen_dir, exist_ok=True)

        for i, gr in enumerate(gen_results):
            if gr["candidate"]["valid"]:
                with open(os.path.join(gen_dir, f"candidate_{i}.py"), "w") as f:
                    f.write(gr["candidate"]["code"])
                with open(os.path.join(gen_dir, f"candidate_{i}_eval.json"), "w") as f:
                    eval_data = json.loads(json.dumps(gr["evaluation"], default=str))
                    json.dump(eval_data, f, indent=2)

        all_history.append({
            "generation": gen + 1,
            "n_valid": sum(1 for c in gen_results if c["evaluation"].get("valid")),
            "n_total": len(gen_results),
            "best_score": best_programs[0]["score"] if best_programs else 0.0,
            "scores": [c["evaluation"].get("overall_score", 0.0) for c in gen_results],
        })

    # =========================================
    # FINAL OUTPUT
    # =========================================
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)

    if best_programs:
        best = best_programs[0]
        final_path = os.path.join(output_dir, "best_underwriter.py")
        with open(final_path, "w") as f:
            f.write(best["code"])
        print(f"\n  Best program score: {best['score']:.3f}")
        print(f"  Saved to: {final_path}")
        print(f"  >>> REVIEW THIS PROGRAM BEFORE DEPLOYMENT <<<")
    else:
        print("\n  No valid programs generated. Check LLM outputs.")

    # Save history
    history_path = os.path.join(output_dir, "evolution_history.json")
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"  History: {history_path}")

    # Save all best programs
    for i, prog in enumerate(best_programs):
        with open(os.path.join(output_dir, f"best_{i+1}_score_{prog['score']:.3f}.py"), "w") as f:
            f.write(prog["code"])

    return best_programs, all_history


def cross_evolve(
    run_config: RunConfig,
    models: list[dict],
    targets_path: str = None,
    n_candidates_per_model: int = 3,
    patience: int = 0,
    convergence_threshold: float = 0.005,
):
    """Run cross-model evolutionary pipeline.

    Pools candidates from multiple models each generation. The best programs
    from any model seed all models in the next generation (island model GA
    with migration every generation).

    Args:
        run_config: Base run configuration (cases, dataset, etc.).
        models: List of {"provider": ..., "model": ...} dicts.
        targets_path: Pre-generated targets JSON (skip target generation).
        n_candidates_per_model: Candidates each model generates per generation.
        patience: Stop after this many gens without improvement. 0 = disabled.
        convergence_threshold: Min score improvement to reset patience counter.
    """
    config = run_config.dataset
    output_dir = run_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations"), exist_ok=True)

    cases_df = pd.read_csv(run_config.cases_csv)

    # Optional holdout from full dataset
    holdout_df = None
    if run_config.full_dataset_csv:
        full_df = pd.read_csv(run_config.full_dataset_csv)
        holdout_df = full_df.sample(
            n=min(run_config.holdout_size, len(full_df)),
            random_state=42,
        )

    # Use provided targets_path or fall back to run_config
    targets_path = targets_path or run_config.targets_path

    # =========================================
    # STEP 1: Generate Targets (or load existing)
    # =========================================
    if targets_path and os.path.exists(targets_path):
        print("=" * 60)
        print("STEP 1: Loading pre-generated targets")
        print("=" * 60)
        with open(targets_path) as f:
            targets = _attach_case_features(json.load(f), cases_df)
        print(f"  Loaded {len(targets)} targets")
    else:
        print("=" * 60)
        print("STEP 1: Generating Targets (LLM reasons about each case)")
        print("=" * 60)
        targets_path = os.path.join(output_dir, "targets.json")
        targets = generate_targets(run_config, output_path=targets_path)

    # Summary
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny
    approve = sum(1 for t in targets if t.get("decision") == approve_str)
    deny = sum(1 for t in targets if t.get("decision") == deny_str)
    print(f"\n  Targets: {approve} {approve_str}, {deny} {deny_str}")

    model_tags = [f"{m['provider']}:{m['model']}" for m in models]
    print(f"\n  Models: {', '.join(model_tags)}")
    print(f"  Candidates per model: {n_candidates_per_model}")
    print(f"  Total candidates per generation: {len(models) * n_candidates_per_model}")

    # =========================================
    # STEP 2-5: Cross-Model Evolutionary Loop
    # =========================================
    print("\n" + "=" * 60)
    print("STEPS 2-5: Cross-Model Evolutionary Program Synthesis")
    print("=" * 60)

    best_programs = []
    all_history = []
    gens_without_improvement = 0
    prev_best_score = 0.0

    max_gen_label = run_config.n_generations if not patience else f"≤{run_config.n_generations}"

    for gen in range(run_config.n_generations):
        print(f"\n--- Generation {gen + 1}/{max_gen_label} ---")

        previous_best = best_programs[:run_config.top_k] if best_programs else None

        # Collect candidates from all models
        all_candidates = []
        for m in models:
            tag = f"{m['provider']}:{m['model']}"
            print(f"\n  Synthesizing from {tag}...")
            candidates = synthesize_programs(
                targets=targets,
                config=config,
                n_candidates=n_candidates_per_model,
                previous_best=previous_best,
                provider=m["provider"],
                model=m["model"],
            )
            for c in candidates:
                c["source_model"] = tag
            all_candidates.extend(candidates)

        # Evaluate ALL candidates together
        gen_results = []
        by_model = {f"{m['provider']}:{m['model']}": {"n_valid": 0, "best": 0.0} for m in models}

        for c in all_candidates:
            tag = c["source_model"]
            if not c["valid"]:
                gen_results.append({"candidate": c, "evaluation": {"overall_score": 0.0, "valid": False}})
                continue

            eval_result = evaluate_candidate(
                code=c["code"],
                targets=targets,
                cases_df=cases_df,
                config=config,
                holdout_df=holdout_df,
            )
            gen_results.append({"candidate": c, "evaluation": eval_result})
            print_evaluation(eval_result, f"{c['candidate_id']} [{tag}]")

            if eval_result.get("valid"):
                by_model[tag]["n_valid"] += 1
                by_model[tag]["best"] = max(by_model[tag]["best"], eval_result["overall_score"])

        # Collect valid results
        valid_results = [
            {
                "code": c["candidate"]["code"],
                "score": c["evaluation"]["overall_score"],
                "source_model": c["candidate"]["source_model"],
                "metrics": c["evaluation"].get("metrics", {}),
                "misclassifications": c["evaluation"].get("misclassifications", []),
            }
            for c in gen_results if c["evaluation"].get("valid")
        ]

        # Clade-based diversity selection instead of pure top-K
        all_programs = best_programs + valid_results
        best_programs = select_diverse_programs(
            all_programs, run_config.top_k, cases_df, config,
        )

        # Print per-model breakdown
        print(f"\n  Per-model results:")
        for tag, stats in by_model.items():
            print(f"    {tag}: {stats['n_valid']} valid, best={stats['best']:.3f}")
        print(f"  All-time best: {best_programs[0]['score']:.3f} [{best_programs[0].get('source_model', '?')}]" if best_programs else "  No valid programs yet")

        # Save generation state
        gen_dir = os.path.join(output_dir, "generations", f"gen_{gen+1:03d}")
        os.makedirs(gen_dir, exist_ok=True)

        for i, gr in enumerate(gen_results):
            if gr["candidate"]["valid"]:
                tag_safe = gr["candidate"]["source_model"].replace(":", "_").replace("/", "_")
                with open(os.path.join(gen_dir, f"candidate_{i}_{tag_safe}.py"), "w") as f:
                    f.write(gr["candidate"]["code"])
                with open(os.path.join(gen_dir, f"candidate_{i}_{tag_safe}_eval.json"), "w") as f:
                    eval_data = json.loads(json.dumps(gr["evaluation"], default=str))
                    json.dump(eval_data, f, indent=2)

        cur_best = best_programs[0]["score"] if best_programs else 0.0
        best_source = best_programs[0].get("source_model", "unknown") if best_programs else "none"
        all_history.append({
            "generation": gen + 1,
            "best_score": cur_best,
            "best_source": best_source,
            "by_model": by_model,
        })

        # Convergence check
        if patience > 0:
            improvement = cur_best - prev_best_score
            if improvement < convergence_threshold:
                gens_without_improvement += 1
                print(f"  Patience: {gens_without_improvement}/{patience} "
                      f"(improvement={improvement:+.4f})")
                if gens_without_improvement >= patience:
                    print(f"  >>> Converged after {gen + 1} generations "
                          f"(no improvement for {patience} gens)")
                    break
            else:
                gens_without_improvement = 0
            prev_best_score = max(prev_best_score, cur_best)

    # =========================================
    # FINAL OUTPUT
    # =========================================
    print("\n" + "=" * 60)
    print("CROSS-MODEL EVOLUTION COMPLETE")
    print("=" * 60)

    if best_programs:
        best = best_programs[0]
        final_path = os.path.join(output_dir, "best_underwriter.py")
        with open(final_path, "w") as f:
            f.write(best["code"])
        print(f"\n  Best program score: {best['score']:.3f}")
        print(f"  Best source model: {best.get('source_model', 'unknown')}")
        print(f"  Saved to: {final_path}")
    else:
        print("\n  No valid programs generated. Check LLM outputs.")

    # Save history
    history_path = os.path.join(output_dir, "evolution_history.json")
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"  History: {history_path}")

    # Save all best programs
    for i, prog in enumerate(best_programs):
        tag_safe = prog.get("source_model", "unknown").replace(":", "_").replace("/", "_")
        with open(os.path.join(output_dir, f"best_{i+1}_score_{prog['score']:.3f}_{tag_safe}.py"), "w") as f:
            f.write(prog["code"])

    return best_programs, all_history


def _discover_program_files(evolve_dir: str) -> list[str]:
    """Find all valid .py candidate files in an evolve output directory."""
    py_files = sorted(glob.glob(os.path.join(evolve_dir, "generations", "gen_*", "*.py")))
    return [p for p in py_files if "_eval" not in p]


def _score_disagreement(
    program_paths: list[str],
    full_df: pd.DataFrame,
    config: DatasetConfig,
) -> np.ndarray:
    """Run all programs on full dataset and compute per-case disagreement.

    For each case: approval_rate = mean(decisions == 1 among valid programs),
    disagreement = min(approval_rate, 1 - approval_rate). Peaks at 0.5
    when programs are evenly split.

    Returns array of shape (n_cases,) with disagreement scores.
    """
    codes = []
    for path in program_paths:
        with open(path) as f:
            codes.append(f.read())

    vectors = []
    for code in tqdm(codes, desc="  Running programs on full dataset"):
        vec = _get_decision_vector(code, full_df, config)
        vectors.append(vec)

    if not vectors:
        return np.zeros(len(full_df))

    matrix = np.stack(vectors)  # (n_programs, n_cases)

    valid_mask = matrix >= 0  # (n_programs, n_cases)
    n_valid = valid_mask.sum(axis=0)  # (n_cases,)

    with np.errstate(invalid="ignore"):
        approve_rate = np.where(
            n_valid > 0,
            (matrix * valid_mask).clip(0).sum(axis=0) / n_valid,
            0.0,
        )

    disagreement = np.minimum(approve_rate, 1 - approve_rate)
    disagreement = np.where(n_valid > 0, disagreement, 0.0)

    return disagreement


def _match_cases_to_full(
    cases_df: pd.DataFrame,
    full_df: pd.DataFrame,
    config: DatasetConfig,
) -> set[int]:
    """Match initial labeled cases to full dataset rows by feature values.

    Returns set of row indices into full_df that correspond to the labeled cases.
    """
    feature_cols = [f.name for f in config.features]
    cols = [c for c in feature_cols if c in cases_df.columns and c in full_df.columns]
    if not cols:
        target_col = config.target.column
        cols = [c for c in cases_df.columns if c in full_df.columns and c != target_col]

    matched = set()
    cases_tuples = set()
    for _, row in cases_df.iterrows():
        cases_tuples.add(tuple(row[cols].values))

    for idx, row in full_df.iterrows():
        if tuple(row[cols].values) in cases_tuples:
            matched.add(idx)

    return matched


def _targets_from_ground_truth(cases_df: pd.DataFrame, config: DatasetConfig, offset: int = 0) -> list[dict]:
    """Construct targets from actual outcomes in the dataset."""
    target_col = config.target.column
    approve_maps_to = config.decisions.approve_maps_to  # "positive" or "negative"
    pos_label = config.target.positive_label
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny

    targets = []
    for i, (_, row) in enumerate(cases_df.iterrows()):
        actual = row[target_col]
        if approve_maps_to == "positive":
            decision = approve_str if actual == pos_label else deny_str
        else:
            decision = deny_str if actual == pos_label else approve_str
        targets.append({
            "case_index": offset + i,
            "decision": decision,
            "confidence": 1.0,
            "reasoning": "ground truth",
            "reason_codes": [],
            "original_features": row.to_dict(),
        })
    return targets


def active_learn(
    run_config: RunConfig,
    models: list[dict],
    targets_path: str,
    full_dataset_path: str,
    max_labels_per_round: int = 50,
    n_rounds: int = 5,
    max_gens_per_round: int = 20,
    n_candidates_per_model: int = 3,
    oracle_provider: str = "anthropic",
    oracle_model: str = "claude-opus-4-6",
    patience: int = 2,
    convergence_threshold: float = 0.005,
    ground_truth: bool = False,
):
    """Active learning loop for evolutionary program synthesis.

    Each round:
    1. Evolve programs until convergence (patience-based early stop)
    2. Score disagreement on full dataset
    3. Select top-disagreement unlabeled cases
    4. Label them (oracle LLM or ground truth)
    5. Expand training set

    Args:
        run_config: Base config (cases_csv = initial labeled cases).
        models: List of {"provider": ..., "model": ...} for cross_evolve.
        targets_path: Path to initial targets JSON (ignored if ground_truth=True).
        full_dataset_path: Path to full dataset CSV.
        max_labels_per_round: Max new cases to label each round.
        n_rounds: Number of active learning rounds.
        max_gens_per_round: Max generations per round (ceiling for convergence).
        n_candidates_per_model: Candidates each model produces per generation.
        oracle_provider: LLM provider for oracle labeling.
        oracle_model: LLM model for oracle labeling.
        patience: Stop evolving after this many gens without improvement.
        convergence_threshold: Min score improvement to reset patience counter.
        ground_truth: Use actual outcomes instead of LLM oracle for labeling.
    """
    config = run_config.dataset
    output_dir = os.path.join(run_config.output_dir, "active_learn")
    os.makedirs(output_dir, exist_ok=True)

    # Load initial state
    cases_df = pd.read_csv(run_config.cases_csv)
    full_df = pd.read_csv(full_dataset_path)

    if ground_truth:
        print("  Using GROUND TRUTH labels (actual outcomes)")
        targets = _targets_from_ground_truth(cases_df, config)
    else:
        with open(targets_path) as f:
            targets = _attach_case_features(json.load(f), cases_df)

    # Match initial cases to full dataset indices
    labeled_indices = _match_cases_to_full(cases_df, full_df, config)
    print(f"Matched {len(labeled_indices)}/{len(cases_df)} initial cases to full dataset "
          f"({len(full_df)} rows)")

    history = []

    for round_num in range(1, n_rounds + 1):
        round_dir = os.path.join(output_dir, f"round_{round_num:03d}")
        os.makedirs(round_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"ACTIVE LEARNING ROUND {round_num}/{n_rounds}")
        print(f"  Labeled cases: {len(cases_df)}, Targets: {len(targets)}")
        print("=" * 70)

        # Save current round's cases and targets
        round_cases_path = os.path.join(round_dir, "cases.csv")
        round_targets_path = os.path.join(round_dir, "targets.json")
        cases_df.to_csv(round_cases_path, index=False)
        with open(round_targets_path, "w") as f:
            json.dump(targets, f, indent=2)

        # ── Step 1: Evolve on current labeled set until convergence ──
        print(f"\n  Step 1: Cross-evolving (up to {max_gens_per_round} gens, "
              f"patience={patience}, "
              f"{len(models)} models x {n_candidates_per_model} candidates)...")

        evolve_dir = os.path.join(round_dir, "evolve")
        evolve_config = copy.copy(run_config)
        evolve_config.cases_csv = round_cases_path
        evolve_config.output_dir = evolve_dir
        evolve_config.n_generations = max_gens_per_round
        evolve_config.full_dataset_csv = full_dataset_path

        best_programs, evolve_history = cross_evolve(
            evolve_config,
            models=models,
            targets_path=round_targets_path,
            n_candidates_per_model=n_candidates_per_model,
            patience=patience,
            convergence_threshold=convergence_threshold,
        )

        best_score = best_programs[0]["score"] if best_programs else 0.0
        print(f"\n  Best program score (on labeled set): {best_score:.3f}")

        # ── Step 2: Discover evolved programs ──
        program_paths = _discover_program_files(evolve_dir)
        print(f"  Step 2: Discovered {len(program_paths)} candidate programs")

        if not program_paths:
            print("  WARNING: No programs found, skipping disagreement scoring")
            history.append({
                "round": round_num,
                "n_labeled": len(cases_df),
                "best_score": best_score,
                "n_programs": 0,
                "top_disagreement": 0.0,
                "new_labels": 0,
            })
            continue

        # ── Step 3: Score disagreement on full dataset ──
        print(f"  Step 3: Scoring disagreement on {len(full_df)} cases...")
        disagreement = _score_disagreement(program_paths, full_df, config)

        # Save disagreement scores
        disagree_df = full_df.copy()
        disagree_df["disagreement"] = disagreement
        disagree_df.to_csv(os.path.join(round_dir, "disagreement.csv"), index=False)

        top_disagree = np.sort(disagreement)[::-1][:10]
        print(f"  Top 10 disagreement scores: {[f'{d:.3f}' for d in top_disagree]}")
        print(f"  Mean disagreement: {disagreement.mean():.3f}")
        print(f"  Cases with disagreement > 0.3: {(disagreement > 0.3).sum()}")

        # ── Step 4: Select unlabeled high-disagreement cases ──
        mask = np.ones(len(full_df), dtype=bool)
        for idx in labeled_indices:
            if 0 <= idx < len(full_df):
                mask[idx] = False

        unlabeled_disagreement = np.where(mask, disagreement, -1.0)
        n_available = mask.sum()
        n_to_select = min(max_labels_per_round, n_available)

        if n_to_select == 0:
            print("  All cases are already labeled!")
            history.append({
                "round": round_num,
                "n_labeled": len(cases_df),
                "best_score": best_score,
                "n_programs": len(program_paths),
                "top_disagreement": float(top_disagree[0]) if len(top_disagree) > 0 else 0.0,
                "new_labels": 0,
            })
            continue

        selected_indices = np.argsort(unlabeled_disagreement)[::-1][:n_to_select]
        selected_indices = [int(i) for i in selected_indices]

        print(f"  Step 4: Selected {len(selected_indices)} high-disagreement cases "
              f"(scores: {unlabeled_disagreement[selected_indices[0]]:.3f} to "
              f"{unlabeled_disagreement[selected_indices[-1]]:.3f})")

        new_cases_df = full_df.iloc[selected_indices].reset_index(drop=True)
        new_cases_path = os.path.join(round_dir, "new_cases.csv")
        new_cases_df.to_csv(new_cases_path, index=False)

        # ── Step 5: Label new cases ──
        if ground_truth:
            print(f"  Step 5: Labeling {len(new_cases_df)} cases from ground truth...")
            new_targets = _targets_from_ground_truth(new_cases_df, config)
            new_targets_path = os.path.join(round_dir, "new_targets.json")
            with open(new_targets_path, "w") as f:
                json.dump(new_targets, f, indent=2)
        else:
            print(f"  Step 5: Oracle labeling {len(new_cases_df)} cases "
                  f"({oracle_provider}:{oracle_model})...")

            oracle_config = copy.copy(run_config)
            oracle_config.cases_csv = new_cases_path
            oracle_config.provider = oracle_provider
            oracle_config.model = oracle_model
            oracle_config.output_dir = round_dir

            new_targets_path = os.path.join(round_dir, "new_targets.json")
            new_targets = generate_targets(oracle_config, output_path=new_targets_path)

        # ── Step 6: Expand training set ──
        offset = len(cases_df)
        for t in new_targets:
            t["case_index"] = offset + t["case_index"]

        targets.extend(new_targets)
        cases_df = pd.concat([cases_df, new_cases_df], ignore_index=True)
        labeled_indices.update(selected_indices)

        print(f"\n  Expanded training set: {len(cases_df)} cases, {len(targets)} targets")

        # ── Step 7: Save round state ──
        round_summary = {
            "round": round_num,
            "n_labeled": len(cases_df),
            "best_score": best_score,
            "n_programs": len(program_paths),
            "top_disagreement": float(top_disagree[0]) if len(top_disagree) > 0 else 0.0,
            "mean_disagreement": float(disagreement.mean()),
            "new_labels": len(new_targets),
        }
        history.append(round_summary)

        with open(os.path.join(round_dir, "round_summary.json"), "w") as f:
            json.dump(round_summary, f, indent=2)

    # ── Final outputs ──
    print("\n" + "=" * 70)
    print("ACTIVE LEARNING COMPLETE")
    print("=" * 70)

    final_cases_path = os.path.join(output_dir, "final_cases.csv")
    final_targets_path = os.path.join(output_dir, "final_targets.json")
    cases_df.to_csv(final_cases_path, index=False)
    with open(final_targets_path, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"  Final labeled cases: {len(cases_df)}")
    print(f"  Final targets: {len(targets)}")
    print(f"  Saved to: {final_cases_path}")
    print(f"  Saved to: {final_targets_path}")

    # Copy best program from last round
    last_round = n_rounds
    last_evolve = os.path.join(output_dir, f"round_{last_round:03d}", "evolve")
    best_path_src = os.path.join(last_evolve, "best_underwriter.py")
    best_path_dst = os.path.join(output_dir, "best_underwriter.py")
    if os.path.exists(best_path_src):
        import shutil
        shutil.copy2(best_path_src, best_path_dst)
        print(f"  Best program: {best_path_dst}")

    # Save history
    history_path = os.path.join(output_dir, "active_learn_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {history_path}")

    # Print progression
    print("\n  Round-by-round progression:")
    for h in history:
        print(f"    Round {h['round']}: {h['n_labeled']} cases, "
              f"best_score={h['best_score']:.3f}, "
              f"top_disagreement={h.get('top_disagreement', 0):.3f}, "
              f"new_labels={h['new_labels']}")

    return cases_df, targets, history
