"""Target generation: LLM reasons through each case and produces decisions."""

import json
import os
import pandas as pd
from tqdm import tqdm

from solon.config.schema import DatasetConfig, RunConfig
from solon.llm.client import call_llm
from solon.prompts.templates import render_target_system_prompt
from solon.prompts.formatters import format_case, build_batch_prompt


def parse_targets(response_text: str) -> list[dict]:
    """Parse the LLM response into structured targets."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        start = text.find("{")
        end = text.rfind("}") + 1
        return [json.loads(text[start:end])]

    return json.loads(text[start:end])


def generate_targets(
    run_config: RunConfig,
    output_path: str = None,
    fewshot_csv: str = None,
    fewshot_meta_json: str = None,
    reasoning_effort: str = None,
    cluster_matched: bool = False,
    clustered_csv: str = None,
):
    """Generate LLM-reasoned targets for the sample cases.

    Args:
        run_config: Full run configuration.
        output_path: Where to save JSON. Defaults to output_dir/targets.json.
        fewshot_csv: Optional few-shot examples CSV.
        fewshot_meta_json: Optional few-shot metadata JSON.
        reasoning_effort: For OpenAI reasoning models.
        cluster_matched: Use cluster-matched few-shot mode.
        clustered_csv: Full clustered dataset CSV.
    """
    config = run_config.dataset
    system_prompt = render_target_system_prompt(config)

    if output_path is None:
        output_path = os.path.join(run_config.output_dir, "targets.json")

    df = pd.read_csv(run_config.cases_csv)
    all_targets = []
    batch_size = run_config.batch_size
    total = len(df)
    n_batches = (total + batch_size - 1) // batch_size

    # Load few-shot examples if provided
    fewshot_text = None
    matcher = None
    if cluster_matched and fewshot_csv and clustered_csv:
        from solon.sampling.selector import ClusterMatcher
        matcher = ClusterMatcher(fewshot_csv, clustered_csv, config)
        print(f"  Cluster-matched mode: each case gets its nearest-cluster examples")
    elif fewshot_csv and fewshot_meta_json:
        fewshot_text = _load_fewshot_examples(fewshot_csv, config)
        print(f"  Loaded few-shot examples from {fewshot_csv}")

    for batch_start in tqdm(range(0, total, batch_size), total=n_batches, desc="Generating targets"):
        batch_end = min(batch_start + batch_size, total)
        batch = [(i, df.iloc[i]) for i in range(batch_start, batch_end)]

        prompt = build_batch_prompt(batch, config, fewshot_text=fewshot_text, cluster_matcher=matcher)

        try:
            response = call_llm(
                system_prompt, prompt,
                provider=run_config.provider,
                model=run_config.model,
                reasoning_effort=reasoning_effort,
            )
            targets = parse_targets(response)

            for j, target in enumerate(targets):
                target["case_index"] = batch_start + j
                target["original_features"] = df.iloc[batch_start + j].to_dict()
                all_targets.append(target)

        except Exception as e:
            print(f"  ERROR on batch {batch_start}-{batch_end}: {e}")
            for j in range(batch_start, batch_end):
                all_targets.append({
                    "case_index": j,
                    "decision": "ERROR",
                    "confidence": 0.0,
                    "reason_codes": [],
                    "reasoning": f"LLM error: {str(e)}",
                    "original_features": df.iloc[j].to_dict(),
                })

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_targets, f, indent=2)

    # Summary
    approve_str = config.decisions.approve
    deny_str = config.decisions.deny
    decisions = [t["decision"] for t in all_targets if t["decision"] != "ERROR"]
    approvals = sum(1 for d in decisions if d == approve_str)
    denials = sum(1 for d in decisions if d == deny_str)
    errors = sum(1 for t in all_targets if t["decision"] == "ERROR")
    print(f"\nTarget Generation Complete:")
    print(f"  Approvals: {approvals}, Denials: {denials}, Errors: {errors}")
    print(f"  Saved to {output_path}")

    return all_targets


def _load_fewshot_examples(fewshot_csv: str, config: DatasetConfig) -> str:
    """Load clustered few-shot examples and format them for the prompt."""
    fewshot_df = pd.read_csv(fewshot_csv)
    target_col = config.target.column
    pos_label = config.target.positive_label
    neg_label = config.target.negative_label
    special = set(config.special_values.values)
    meanings = config.special_values.meanings

    parts = [
        "REFERENCE CASES WITH KNOWN OUTCOMES:",
        "Below are real applicant profiles from across the risk spectrum, with their actual outcomes.",
        "Use these to calibrate your judgment — they illustrate the range of profiles.\n",
    ]

    for i, (_, row) in enumerate(fewshot_df.iterrows()):
        if target_col in row.index:
            outcome = config.target.negative_meaning if row[target_col] == neg_label else config.target.positive_meaning
        else:
            outcome = "unknown"
        parts.append(f"--- REFERENCE CASE {i+1} (Outcome: {outcome.upper()}) ---")
        for col, val in row.items():
            if col in (target_col, "cluster"):
                continue
            if val in special:
                label = meanings.get(int(val), "special")
                parts.append(f"  {col}: {label}")
            else:
                parts.append(f"  {col}: {val}")
        parts.append("")

    return "\n".join(parts)
