"""Program synthesis: LLM generates deterministic Python functions.

The output is auditable code — not a model, not weights.
"""

import importlib.util
import re
import pandas as pd
import numpy as np

from solon.config.schema import DatasetConfig
from solon.llm.client import call_llm
from solon.prompts.templates import render_synthesis_system_prompt
from solon.prompts.formatters import build_synthesis_prompt


def extract_function(response_text: str) -> str:
    """Extract the Python function from LLM response."""
    text = response_text.strip()
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    if not text.startswith("import") and not text.startswith("def "):
        match = re.search(r'((?:import.*\n)*\s*def \w+\(.*?\).*)', text, re.DOTALL)
        if match:
            text = match.group(1)
    return text


def validate_program(code: str, config: DatasetConfig) -> tuple[bool, str]:
    """Check if the program is syntactically valid and safe."""
    dangerous = ["os.system", "subprocess", "eval(", "exec(", "__import__",
                 "open(", "file(", "input(", "raw_input"]
    for d in dangerous:
        if d in code:
            return False, f"Dangerous operation found: {d}"

    # Reject programs that reference the target variable (data leakage)
    target_col = config.target.column
    if target_col in code:
        return False, f"References target variable {target_col} (data leakage)"

    try:
        compile(code, "<synthesized>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, "OK"


def load_program(code: str):
    """Load a synthesized program and return the main function."""
    temp_path = "/tmp/_synthesized_underwriter.py"
    with open(temp_path, "w") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location("_synth", temp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Try to find the function - look for common names
    for name in ("underwrite", "evaluate", "decide"):
        if hasattr(mod, name):
            return getattr(mod, name)

    # Fallback: find any function that takes a dict arg
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if callable(attr) and not attr_name.startswith("_"):
            return attr

    raise AttributeError("No callable function found in synthesized code")


def run_program(fn, cases_df: pd.DataFrame, config: DatasetConfig) -> list[dict]:
    """Run the synthesized program on all cases. Returns list of results."""
    target_col = config.target.column
    drop_cols = [c for c in (target_col,) if c in cases_df.columns]
    clean_df = cases_df.drop(columns=drop_cols) if drop_cols else cases_df

    results = []
    for _, row in clean_df.iterrows():
        app = row.to_dict()
        try:
            result = fn(app)
            results.append({
                "decision": result.get("decision", "ERROR"),
                "reason_codes": result.get("reason_codes", []),
                "score": result.get("score", 0.0),
                "error": None,
            })
        except Exception as e:
            results.append({
                "decision": "ERROR",
                "reason_codes": [],
                "score": 0.0,
                "error": str(e),
            })
    return results


def synthesize_programs(
    targets: list[dict],
    config: DatasetConfig,
    n_candidates: int = 5,
    previous_best: list[dict] = None,
    provider: str = "anthropic",
    model: str = None,
) -> list[dict]:
    """Generate n candidate programs.

    Returns list of {code, valid, error, candidate_id} dicts.
    """
    system_prompt = render_synthesis_system_prompt(config)
    user_prompt = build_synthesis_prompt(targets, config, previous_best)
    candidates = []

    for i in range(n_candidates):
        print(f"  Generating candidate {i+1}/{n_candidates}...")
        try:
            response = call_llm(system_prompt, user_prompt, provider=provider, model=model)
            code = extract_function(response)
            valid, error = validate_program(code, config)

            candidates.append({
                "code": code,
                "valid": valid,
                "validation_error": error if not valid else None,
                "candidate_id": i,
            })

            if valid:
                print(f"    Valid program generated")
            else:
                print(f"    Invalid: {error}")

        except Exception as e:
            candidates.append({
                "code": "",
                "valid": False,
                "validation_error": str(e),
                "candidate_id": i,
            })
            print(f"    Error: {e}")

    return candidates
