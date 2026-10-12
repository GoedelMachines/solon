"""Format cases and batches for LLM prompts, parameterized by DatasetConfig."""

import pandas as pd
from solon.config.schema import DatasetConfig


def format_case(row: pd.Series, config: DatasetConfig) -> str:
    """Format a single application as readable text for the LLM."""
    lines = ["APPLICATION DATA:"]
    special = set(config.special_values.values)
    meanings = config.special_values.meanings
    target_col = config.target.column

    for col, val in row.items():
        if col in (target_col, "cluster"):
            continue
        if val in special:
            label = meanings.get(int(val), "special")
            lines.append(f"  {col}: {label}")
        else:
            lines.append(f"  {col}: {val}")
    return "\n".join(lines)


def build_batch_prompt(
    cases: list[tuple[int, pd.Series]],
    config: DatasetConfig,
    fewshot_text: str = None,
    cluster_matcher=None,
) -> str:
    """Build a prompt for a batch of cases.

    Modes:
    - cluster_matcher: each case gets its own matched reference examples inline
    - fewshot_text: all examples prepended (legacy mode)
    - neither: no few-shot
    """
    parts = []
    product = config.product_context.strip() if config.product_context else "application"

    if cluster_matcher:
        parts.append(
            f"Evaluate the following {product} applications. Each application is shown with "
            "similar reference cases that have known outcomes. Use these matched references "
            "to calibrate your judgment — they are the most similar profiles from our data.\n"
        )
        for _, (idx, row) in enumerate(cases):
            parts.append(f"--- APPLICATION {idx} ---")
            cluster_id = cluster_matcher.nearest_cluster(row)
            parts.append(cluster_matcher.format_matched_examples(cluster_id))
            parts.append(format_case(row, config))
            parts.append("")
    elif fewshot_text:
        parts.append(fewshot_text)
        parts.append("\n" + "=" * 60 + "\n")
        parts.append("Now evaluate the following NEW applications. For each, provide your decision as JSON.\n")
        for _, (idx, row) in enumerate(cases):
            parts.append(f"--- APPLICATION {idx} ---")
            parts.append(format_case(row, config))
            parts.append("")
    else:
        parts.append(f"Evaluate the following applications. For each, provide your decision as JSON.\n")
        for _, (idx, row) in enumerate(cases):
            parts.append(f"--- APPLICATION {idx} ---")
            parts.append(format_case(row, config))
            parts.append("")

    approve = config.decisions.approve
    deny = config.decisions.deny
    parts.append(
        "Respond with a JSON array containing one object per application, in order. "
        f"Each object must have: application_id (int), decision ({approve}/{deny}), "
        "confidence (float), reason_codes (list of strings), reasoning (string)."
    )
    return "\n".join(parts)


def build_synthesis_prompt(
    targets: list[dict],
    config: DatasetConfig,
    previous_best: list[dict] = None,
) -> str:
    """Build the prompt for program synthesis with best-shot examples and eval feedback."""
    parts = []
    summary_features = config.summary_features

    # Show target examples
    parts.append("Here are example applications with the CORRECT decision:\n")
    for t in targets[:20]:
        feats = t.get("original_features", {})
        feat_strs = []
        for sf in summary_features[:5]:
            feat_strs.append(f"{sf}={feats.get(sf, '?')}")
        parts.append(f"Application ({', '.join(feat_strs)})")
        parts.append(f"  -> {t['decision']} | Reasons: {t.get('reason_codes', [])}")
        parts.append(f"  -> Rationale: {t.get('reasoning', 'N/A')[:150]}")
        parts.append("")

    approve_count = sum(1 for t in targets if t.get("decision") == config.decisions.approve)
    deny_count = sum(1 for t in targets if t.get("decision") == config.decisions.deny)
    parts.append(f"\nOverall: {approve_count} approvals, {deny_count} denials out of {len(targets)} cases.")

    # Best-shot: show previous best programs with evaluation feedback
    if previous_best:
        parts.append("\n\n--- PREVIOUS BEST PROGRAMS WITH EVALUATION FEEDBACK ---")
        parts.append("Study what each program got RIGHT and WRONG. Fix the mistakes.\n")
        for i, prog_data in enumerate(previous_best):
            code = prog_data["code"]
            metrics = prog_data.get("metrics", {})
            misses = prog_data.get("misclassifications", [])

            parts.append(f"### Program {i+1} (score: {prog_data.get('score', 0):.3f})")
            parts.append(f"Accuracy: {metrics.get('accuracy', 0):.1%} "
                         f"({metrics.get('correct', 0)}/{metrics.get('total_compared', 0)})")
            parts.append(f"Simplicity: {metrics.get('simplicity', 0):.2f} "
                         f"({metrics.get('n_conditions', '?')} conditions)")
            parts.append(f"Robustness: {metrics.get('robustness', 0):.2f}")
            parts.append(f"Coverage: {metrics.get('coverage', 0):.1%}")

            if misses:
                parts.append(f"\nMISCLASSIFIED CASES ({len(misses)} errors):")
                for m in misses[:15]:
                    f = m["features"]
                    parts.append(f"  Case {m['case_index']}: predicted {m['predicted']}, "
                                 f"should be {m['expected']}")
                    feat_strs = []
                    for sf in summary_features:
                        feat_strs.append(f"{sf}={f.get(sf, '?')}")
                    parts.append(f"    {', '.join(feat_strs)}")
                if len(misses) > 15:
                    parts.append(f"  ... and {len(misses) - 15} more errors")
            else:
                parts.append("\nNo misclassifications — perfect accuracy!")

            parts.append(f"\nCode:")
            parts.append(code)
            parts.append("")

    func_name = config.synthesizer.function_name
    parts.append(f"\nWrite the COMPLETE {func_name}() function. "
                 "Return ONLY the Python code, no explanation. "
                 "The function must match the decisions above as closely as possible. "
                 "Pay special attention to fixing the misclassified cases from previous programs.")

    return "\n".join(parts)
