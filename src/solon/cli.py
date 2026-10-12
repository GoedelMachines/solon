"""CLI entry point for solon.

Subcommands:
  sample           Sample representative cases from full dataset
  cluster          Cluster full dataset and select few-shot examples
  targets          Generate LLM-reasoned targets for sample cases
  eval-targets     Evaluate targets against actual outcomes
  synthesize       Synthesize candidate programs from targets
  eval-program     Evaluate a synthesized program against actual outcomes
  evolve           Run full evolutionary pipeline (targets → synthesize → eval → repeat)
  cross-evolve     Cross-model evolutionary pipeline (pool candidates from multiple models)
  active-learn     Active learning loop (evolve → disagreement → oracle label → expand)
  analyze          Post-hoc analysis of evolve runs:
    compare            evaluate every candidate on the full dataset + code analysis
    similarity         behavioral + code similarity across candidate programs
    decision-boundary  2D decision-boundary landscape for evolved programs
"""

import argparse
import sys

from solon.config.loader import build_run_config, load_dataset_config


def _add_common_args(parser):
    """Add args shared across subcommands."""
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset YAML config (e.g., datasets/heloc/dataset.yaml)",
    )
    parser.add_argument(
        "--cases", required=True,
        help="Path to sample cases CSV",
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Output directory (default: outputs)",
    )


def _add_llm_args(parser):
    """Add LLM provider/model args."""
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "openai", "gemini", "deepseek", "together", "openrouter"],
        help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="Model name (default: provider default)")


# ── sample ───────────────────────────────────────────────────────────────

def cmd_sample(args):
    """Sample representative cases from the full dataset."""
    import pandas as pd
    from solon.sampling.selector import select_cases

    config = load_dataset_config(args.dataset)
    df = pd.read_csv(args.input)

    sample = select_cases(
        df, config,
        n_cases=args.n,
        n_clusters=args.clusters,
        random_state=args.seed,
    )

    sample.to_csv(args.output_file, index=False)
    target_col = config.target.column
    dist = sample[target_col].value_counts().to_dict()
    print(f"Sampled {len(sample)} cases -> {args.output_file}")
    print(f"  Target distribution: {dist}")


# ── cluster ──────────────────────────────────────────────────────────────

def cmd_cluster(args):
    """Cluster full dataset and select few-shot examples."""
    import pandas as pd
    from solon.sampling.cluster_kmeans import cluster_dataset

    config = load_dataset_config(args.dataset)
    df = pd.read_csv(args.input)

    clustered_df, fewshot_df, profiles, selected = cluster_dataset(
        df, config,
        output_dir=args.output,
        min_k=args.min_k,
        max_k=args.max_k,
    )

    print(f"\nFew-shot examples: {len(fewshot_df)} cases")
    print(f"Clusters: {len(profiles)}")
    for p in profiles:
        print(f"  Cluster {p['cluster']}: n={p['n']} ({p['pct']:.1f}%), bad_rate={p['bad_rate']:.1%}")


# ── targets ──────────────────────────────────────────────────────────────

def cmd_targets(args):
    """Generate LLM-reasoned targets."""
    import os
    from solon.pipeline.target_generator import generate_targets

    run_config = build_run_config(
        args.dataset,
        cases_csv=args.cases,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
    )

    output_path = args.output_file
    if output_path is None:
        os.makedirs(args.output, exist_ok=True)
        model_tag = args.model or run_config.provider
        model_tag = model_tag.replace("/", "_").replace(":", "_")
        output_path = os.path.join(args.output, f"targets_{model_tag}.json")

    targets = generate_targets(
        run_config,
        output_path=output_path,
        fewshot_csv=args.fewshot,
        fewshot_meta_json=args.fewshot_meta,
        reasoning_effort=args.reasoning_effort,
        cluster_matched=args.cluster_matched,
        clustered_csv=args.clustered,
    )
    print(f"\nSaved {len(targets)} targets to {output_path}")


# ── eval-targets ─────────────────────────────────────────────────────────

def cmd_eval_targets(args):
    """Evaluate targets against actual outcomes."""
    import json
    import pandas as pd
    from solon.analysis.compare_targets import (
        score_targets, print_model_report, print_comparison_table,
    )

    config = load_dataset_config(args.dataset)
    cases = pd.read_csv(args.cases)
    all_scores = {}

    for targets_file in args.targets:
        with open(targets_file) as f:
            targets = json.load(f)
        name = targets_file.rsplit("/", 1)[-1].replace("targets_", "").replace(".json", "")
        scores = score_targets(targets, cases, config)
        all_scores[name] = scores
        print_model_report(name, scores)

    if len(all_scores) > 1:
        print_comparison_table(all_scores)


# ── synthesize ───────────────────────────────────────────────────────────

def cmd_synthesize(args):
    """Synthesize candidate programs from targets."""
    import json
    import os
    from solon.pipeline.program_synthesizer import synthesize_programs, validate_program

    config = load_dataset_config(args.dataset)
    with open(args.targets) as f:
        targets = json.load(f)
    if args.cases:
        import pandas as pd
        from solon.pipeline.evolve import _attach_case_features
        targets = _attach_case_features(targets, pd.read_csv(args.cases))

    previous_best = None
    if args.seed_programs:
        previous_best = []
        for path in args.seed_programs:
            with open(path) as f:
                previous_best.append({"code": f.read(), "score": 0.0})

    candidates = synthesize_programs(
        targets=targets,
        config=config,
        n_candidates=args.candidates,
        previous_best=previous_best,
        provider=args.provider,
        model=args.model,
    )

    os.makedirs(args.output, exist_ok=True)
    for i, c in enumerate(candidates):
        if c["valid"]:
            path = os.path.join(args.output, f"candidate_{i}.py")
            with open(path, "w") as f:
                f.write(c["code"])
            print(f"  candidate_{i}.py  (valid)")
        else:
            print(f"  candidate_{i}  INVALID: {c['validation_error']}")

    valid_count = sum(1 for c in candidates if c["valid"])
    print(f"\n{valid_count}/{len(candidates)} valid candidates saved to {args.output}")


# ── eval-program ─────────────────────────────────────────────────────────

def cmd_eval_program(args):
    """Evaluate a synthesized program against actual outcomes."""
    from solon.analysis.eval_program import evaluate

    config = load_dataset_config(args.dataset)
    for program_path in args.programs:
        evaluate(program_path, args.cases, config)


# ── compare-programs ─────────────────────────────────────────────────────

def cmd_compare_programs(args):
    """Compare all candidates across evolve output dirs."""
    import glob
    import os
    import pandas as pd
    from solon.analysis.analyze_candidates import (
        evaluate_candidate_against_actuals, analyze_code_structure,
    )

    config = load_dataset_config(args.dataset)
    cases = pd.read_csv(args.cases)

    # Discover all .py candidates in given dirs
    all_programs = []
    for d in args.dirs:
        py_files = sorted(glob.glob(os.path.join(d, "generations", "gen_*", "*.py")))
        py_files = [p for p in py_files if "_eval" not in p]
        run_name = os.path.basename(d.rstrip("/"))
        for p in py_files:
            gen = p.split("/gen_")[1].split("/")[0] if "/gen_" in p else "?"
            cand = os.path.basename(p).replace(".py", "")
            all_programs.append((run_name, gen, cand, p))

    if not all_programs:
        print("No candidate programs found.")
        return

    print(f"Evaluating {len(all_programs)} candidates against {len(cases)} cases...\n")

    header = f"{'Run':<25} {'Gen':>3} {'Candidate':<45} {'Lines':>5} {'Conds':>5} {'Feats':>5}  {'Acc':>6} {'F1':>6} {'Appr%':>6}"
    print(header)
    print("-" * len(header))

    best_acc = (0, None)
    best_f1 = (0, None)

    for run_name, gen, cand, path in all_programs:
        with open(path) as f:
            code = f.read()
        structure = analyze_code_structure(code)
        metrics = evaluate_candidate_against_actuals(path, cases, config)

        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        appr = metrics["approve_rate"]

        if acc > best_acc[0]:
            best_acc = (acc, f"{run_name} gen{gen} {cand}")
        if f1 > best_f1[0]:
            best_f1 = (f1, f"{run_name} gen{gen} {cand}")

        print(
            f"{run_name:<25} {gen:>3} {cand:<45} "
            f"{structure['lines']:>5} {structure['n_conditions']:>5} {structure['n_features']:>5}  "
            f"{acc:>6.3f} {f1:>6.3f} {appr:>6.3f}"
        )

    print(f"\nBest accuracy: {best_acc[0]:.3f}  ({best_acc[1]})")
    print(f"Best F1:       {best_f1[0]:.3f}  ({best_f1[1]})")


# ── similarity ──────────────────────────────────────────────────────────

def cmd_similarity(args):
    """Analyze behavioral + code similarity across candidate programs."""
    import pandas as pd
    from solon.analysis.program_similarity import run_similarity_analysis

    config = load_dataset_config(args.dataset)
    cases = pd.read_csv(args.cases)
    run_similarity_analysis(
        dirs=args.dirs,
        cases_df=cases,
        config=config,
        output_dir=args.output,
    )


# ── evolve ───────────────────────────────────────────────────────────────

def cmd_evolve(args):
    """Run the full evolutionary pipeline."""
    from solon.pipeline.evolve import evolve

    run_config = build_run_config(
        args.dataset,
        cases_csv=args.cases,
        full_dataset_csv=args.full_dataset,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        n_candidates=args.candidates,
        n_generations=args.generations,
        top_k=args.top_k,
        targets_path=args.targets,
        batch_size=args.batch_size,
        holdout_size=args.holdout_size,
    )
    evolve(run_config)


# ── cross-evolve ────────────────────────────────────────────────────

def _parse_model_spec(spec: str) -> dict:
    """Parse 'provider:model' string into a dict."""
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid model spec '{spec}'. Expected format: provider:model "
            f"(e.g., gemini:gemini-3-pro-preview)"
        )
    return {"provider": parts[0], "model": parts[1]}


def cmd_cross_evolve(args):
    """Run cross-model evolutionary pipeline."""
    from solon.pipeline.evolve import cross_evolve

    models = [_parse_model_spec(s) for s in args.models]

    run_config = build_run_config(
        args.dataset,
        cases_csv=args.cases,
        full_dataset_csv=args.full_dataset,
        output_dir=args.output,
        n_generations=args.generations,
        top_k=args.top_k,
        targets_path=args.targets,
        batch_size=args.batch_size,
        holdout_size=args.holdout_size,
    )
    cross_evolve(
        run_config,
        models=models,
        n_candidates_per_model=args.candidates_per_model,
    )


# ── decision-boundary ─────────────────────────────────────────────────

def cmd_decision_boundary(args):
    """Plot decision boundary landscape for evolved programs."""
    import pandas as pd
    from solon.analysis.decision_boundary import (
        discover_program_files, compute_consensus_and_disagreement,
        tsne_reduce, plot_decision_landscape,
    )
    import numpy as np

    config = load_dataset_config(args.dataset)
    full_df = pd.read_csv(args.cases)

    program_paths = discover_program_files(args.dirs)
    if not program_paths:
        print("No programs found.")
        return

    print(f"Found {len(program_paths)} programs across {len(args.dirs)} directories")
    print(f"Dataset: {len(full_df)} cases")

    print("Computing program consensus and disagreement...")
    consensus, disagreement = compute_consensus_and_disagreement(
        program_paths, full_df, config,
    )

    print("t-SNE reducing features to 2D...")
    coords = tsne_reduce(full_df, config)

    # Optional: labeled cases mask
    labeled_mask = None
    if args.labeled_cases:
        labeled_df = pd.read_csv(args.labeled_cases)
        from solon.pipeline.evolve import _match_cases_to_full
        labeled_indices = _match_cases_to_full(labeled_df, full_df, config)
        labeled_mask = np.zeros(len(full_df), dtype=bool)
        for idx in labeled_indices:
            if 0 <= idx < len(full_df):
                labeled_mask[idx] = True
        print(f"Marked {labeled_mask.sum()} labeled cases")

    # Actual outcomes — map to approve=1, deny=0 to match consensus panel colors
    target_col = config.target.column
    actual = None
    if target_col in full_df.columns:
        pos_label = config.target.positive_label
        approve_maps_to = config.decisions.approve_maps_to
        raw = full_df[target_col].values
        if approve_maps_to == "positive":
            actual = (raw == pos_label).astype(float)
        else:
            actual = (raw != pos_label).astype(float)

    plot_decision_landscape(
        coords, consensus, disagreement,
        output_path=args.output_file,
        labeled_mask=labeled_mask,
        actual_outcomes=actual,
        title_prefix=args.title + ": " if args.title else "",
    )

    # Print stats
    print(f"\nStats:")
    print(f"  Mean consensus: {consensus.mean():.3f}")
    print(f"  Mean disagreement: {disagreement.mean():.3f}")
    print(f"  High disagreement (>0.3): {(disagreement > 0.3).sum()}")
    print(f"  Approve-leaning (consensus > 0.5): {(consensus > 0.5).sum()}")
    print(f"  Deny-leaning (consensus < 0.5): {(consensus < 0.5).sum()}")


# ── active-learn ─────────────────────────────────────────────────────

def cmd_active_learn(args):
    """Run active learning loop."""
    from solon.pipeline.evolve import active_learn

    models = [_parse_model_spec(s) for s in args.models]

    run_config = build_run_config(
        args.dataset,
        cases_csv=args.cases,
        full_dataset_csv=args.full_dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        holdout_size=args.holdout_size,
        top_k=args.top_k,
    )
    active_learn(
        run_config,
        models=models,
        targets_path=args.targets,
        full_dataset_path=args.full_dataset,
        max_labels_per_round=args.max_labels_per_round,
        n_rounds=args.rounds,
        max_gens_per_round=args.max_gens_per_round,
        n_candidates_per_model=args.candidates_per_model,
        oracle_provider=args.oracle_provider,
        oracle_model=args.oracle_model,
        patience=args.patience,
        convergence_threshold=args.convergence_threshold,
        ground_truth=args.ground_truth,
    )


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="solon",
        description="Oracle-Guided Evolutionary Program Synthesis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── sample
    p = sub.add_parser("sample", help="Sample representative cases from full dataset")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--input", required=True, help="Full dataset CSV")
    p.add_argument("--output-file", default="data/sample_cases.csv", help="Output CSV path")
    p.add_argument("-n", type=int, default=100, help="Number of cases to sample (default: 100)")
    p.add_argument("--clusters", type=int, default=20, help="Number of clusters for stratified sampling")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.set_defaults(func=cmd_sample)

    # ── cluster
    p = sub.add_parser("cluster", help="Cluster dataset and select few-shot examples")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--input", required=True, help="Full dataset CSV")
    p.add_argument("--output", default="outputs/clustering", help="Output directory")
    p.add_argument("--min-k", type=int, default=4, help="Min clusters to try (default: 4)")
    p.add_argument("--max-k", type=int, default=15, help="Max clusters to try (default: 15)")
    p.set_defaults(func=cmd_cluster)

    # ── targets
    p = sub.add_parser("targets", help="Generate LLM-reasoned targets")
    _add_common_args(p)
    _add_llm_args(p)
    p.add_argument("--batch-size", type=int, default=None, help="Cases per LLM call")
    p.add_argument("--output-file", default=None, help="Output JSON path (default: auto-named in output dir)")
    p.add_argument("--reasoning-effort", default=None, help="Reasoning effort for OpenAI models (low/medium/high)")
    p.add_argument("--fewshot", default=None, help="Few-shot examples CSV (from cluster command)")
    p.add_argument("--fewshot-meta", default=None, help="Few-shot metadata JSON (from cluster command)")
    p.add_argument("--cluster-matched", action="store_true", help="Use per-case cluster-matched few-shot mode")
    p.add_argument("--clustered", default=None, help="Clustered dataset CSV (required with --cluster-matched)")
    p.set_defaults(func=cmd_targets)

    # ── eval-targets
    p = sub.add_parser("eval-targets", help="Evaluate targets against actual outcomes")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--cases", required=True, help="Path to cases CSV (with actual outcomes)")
    p.add_argument("targets", nargs="+", help="One or more targets JSON files to evaluate")
    p.set_defaults(func=cmd_eval_targets)

    # ── synthesize
    p = sub.add_parser("synthesize", help="Synthesize candidate programs from targets")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--targets", required=True, help="Targets JSON file")
    p.add_argument("--cases", default=None, help="Cases CSV; fills features missing from the targets (e.g. shipped examples)")
    p.add_argument("--output", default="outputs/candidates", help="Output directory for candidates")
    _add_llm_args(p)
    p.add_argument("--candidates", type=int, default=5, help="Number of candidates to generate")
    p.add_argument("--seed-programs", nargs="*", help="Previous best programs to use as few-shot examples")
    p.set_defaults(func=cmd_synthesize)

    # ── eval-program
    p = sub.add_parser("eval-program", help="Evaluate program(s) against actual outcomes")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--cases", required=True, help="Path to cases CSV (with actual outcomes)")
    p.add_argument("programs", nargs="+", help="One or more .py program files to evaluate")
    p.set_defaults(func=cmd_eval_program)

    # ── evolve
    p = sub.add_parser("evolve", help="Run full evolutionary pipeline")
    _add_common_args(p)
    _add_llm_args(p)
    p.add_argument("--full-dataset", default=None, help="Full dataset CSV (for holdout testing)")
    p.add_argument("--candidates", type=int, default=None, help="Candidates per generation")
    p.add_argument("--generations", type=int, default=None, help="Number of generations")
    p.add_argument("--top-k", type=int, default=None, help="Programs kept per generation")
    p.add_argument("--targets", default=None, help="Pre-generated targets JSON (skip target generation)")
    p.add_argument("--batch-size", type=int, default=None, help="Cases per LLM call")
    p.add_argument("--holdout-size", type=int, default=None, help="Holdout sample size")
    p.set_defaults(func=cmd_evolve)

    # ── cross-evolve
    p = sub.add_parser("cross-evolve", help="Cross-model evolutionary pipeline")
    _add_common_args(p)
    p.add_argument(
        "--models", required=True, nargs="+",
        help="Models as provider:model (e.g., gemini:gemini-3-pro-preview anthropic:claude-opus-4-6)",
    )
    p.add_argument("--full-dataset", default=None, help="Full dataset CSV (for holdout testing)")
    p.add_argument("--candidates-per-model", type=int, default=3, help="Candidates per model per generation (default: 3)")
    p.add_argument("--generations", type=int, default=None, help="Number of generations")
    p.add_argument("--top-k", type=int, default=None, help="Programs kept per generation")
    p.add_argument("--targets", default=None, help="Pre-generated targets JSON (skip target generation)")
    p.add_argument("--batch-size", type=int, default=None, help="Cases per LLM call")
    p.add_argument("--holdout-size", type=int, default=None, help="Holdout sample size")
    p.set_defaults(func=cmd_cross_evolve)

    # ── analyze: post-hoc analysis of evolve runs (not part of the main path)
    analyze = sub.add_parser("analyze", help="Analyze evolve runs: compare, similarity, decision-boundary")
    asub = analyze.add_subparsers(dest="analysis", required=True)

    # ── analyze compare
    p = asub.add_parser("compare", help="Evaluate every candidate across evolve runs on the full dataset")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--cases", required=True, help="Path to cases CSV (with actual outcomes)")
    p.add_argument("dirs", nargs="+", help="One or more evolve output directories to scan")
    p.set_defaults(func=cmd_compare_programs)

    # ── analyze similarity
    p = asub.add_parser("similarity", help="Analyze behavioral + code similarity across programs")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--cases", required=True, help="Path to cases CSV (with actual outcomes)")
    p.add_argument("--output", default="outputs/similarity", help="Output directory")
    p.add_argument("dirs", nargs="+", help="One or more evolve output directories to scan")
    p.set_defaults(func=cmd_similarity)

    # ── analyze decision-boundary
    p = asub.add_parser("decision-boundary", help="Plot 2D decision boundary landscape for evolved programs")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML config")
    p.add_argument("--cases", required=True, help="Full dataset CSV (used for t-SNE + program evaluation)")
    p.add_argument("--labeled-cases", default=None, help="Labeled cases CSV (to highlight on plot)")
    p.add_argument("--output-file", default="decision_landscape.png", help="Output PNG path (default: decision_landscape.png)")
    p.add_argument("--title", default="", help="Title prefix for the plot")
    p.add_argument("dirs", nargs="+", help="One or more evolve output directories containing programs")
    p.set_defaults(func=cmd_decision_boundary)

    # ── active-learn
    p = sub.add_parser("active-learn", help="Active learning loop (evolve → disagreement → oracle label → expand)")
    _add_common_args(p)
    p.add_argument(
        "--models", required=True, nargs="+",
        help="Models as provider:model (e.g., gemini:gemini-3-pro-preview anthropic:claude-opus-4-6)",
    )
    p.add_argument("--full-dataset", required=True, help="Full dataset CSV (for disagreement scoring)")
    p.add_argument("--targets", default=None, help="Pre-generated targets JSON for initial labeled cases")
    p.add_argument("--ground-truth", action="store_true", help="Use actual outcomes instead of LLM oracle for labeling")
    p.add_argument("--oracle-provider", default="anthropic", help="LLM provider for oracle labeling (default: anthropic)")
    p.add_argument("--oracle-model", default="claude-opus-4-6", help="LLM model for oracle labeling (default: claude-opus-4-6)")
    p.add_argument("--max-labels-per-round", type=int, default=50, help="Max new cases to label per round (default: 50)")
    p.add_argument("--rounds", type=int, default=5, help="Number of active learning rounds (default: 5)")
    p.add_argument("--max-gens-per-round", type=int, default=20, help="Max evolution generations per round (default: 20)")
    p.add_argument("--patience", type=int, default=2, help="Stop evolving after N gens without improvement (default: 2)")
    p.add_argument("--convergence-threshold", type=float, default=0.005, help="Min score improvement to reset patience (default: 0.005)")
    p.add_argument("--candidates-per-model", type=int, default=3, help="Candidates per model per generation (default: 3)")
    p.add_argument("--top-k", type=int, default=None, help="Programs kept per generation")
    p.add_argument("--batch-size", type=int, default=None, help="Cases per LLM call")
    p.add_argument("--holdout-size", type=int, default=None, help="Holdout sample size")
    p.set_defaults(func=cmd_active_learn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
