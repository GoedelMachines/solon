# Solon

Distillation of LLM reasoning into deterministic code.

*Named after Solon, the Athenian lawgiver who wrote the law down where anyone could read it.*

An LLM reasons through a small sample of cases (the *targets*: decision + reason codes), then
writes plain `if/elif/else` programs that reproduce those decisions. Candidates are scored,
the best seed the next generation (FunSearch-style), and the output is a function like
`underwrite(app) -> {decision, reason_codes, score}`. **The LLM is used at synthesis time only**:
the shipped artifact has no weights and makes no model call, so the same input always gives the
same output, and a reviewer can read every rule.

Status: research code, `0.1.0`, no stability guarantees. One worked dataset (FICO HELOC credit
underwriting) with example outputs; a Lending Club config is in progress (`docs/plans/`).

## Results — read before using

On FICO HELOC, the best evolved rule program (written from 100 LLM-labelled cases) is
**1.3 AUC points behind XGBoost trained on the same 100 cases, and 5.2 points behind XGBoost
trained on all 10,459 rows.** It is not a replacement for a well-tuned classical model; use it
when labels are scarce, the decision boundary can be put into words, or every rule must be
readable.

| HELOC | AUC | Accuracy |
|---|---|---|
| Best evolved program, `examples/heloc/cross_evolve/best_underwriter.py`, on all 10,459 rows | 0.743 | 0.707 |
| XGBoost trained on the same 100 cases, scored on the 9,775 rows not identical to one of them | 0.756 ± 0.004 | — |
| XGBoost on all rows, 5-fold CV | 0.795 ± 0.011 | — |

Reproduce: `python datasets/heloc/fetch_heloc.py`, then
`solon eval-program --dataset datasets/heloc/dataset.yaml --cases data/heloc_real.csv examples/heloc/cross_evolve/best_underwriter.py`
(the program's AUC is from its `score` output; it also reports accuracy, F1 and the decision-only
AUC, 0.705), and `python datasets/heloc/baselines.py` (needs `pip install xgboost`). The program's
row includes the 100 cases it was evolved on (1% of the data). The evolution runs are single-seed.
Per-model runs: [`datasets/heloc/results.md`](datasets/heloc/results.md).

**Not reproducible here:** an earlier CFPB-complaints experiment (rule program + case base +
Bayesian network, rank-blended: 0.930 AUC against 0.920 for target-encoded XGBoost) used
scripts that are not in this repository. Treat it as unverified.

## Pipeline

```
dataset CSV + dataset YAML (features, regulatory context, decision framework)
  │  sample: stratified k-means → ~100 cases
  ▼
targets     LLM decides each case → APPROVE/DENY + reason codes + reasoning   [human review]
  ▼
synthesize  LLM writes a deterministic Python function; rejected if it uses unsafe calls
            (os.system, subprocess, eval, exec, open, …) or reads the target column
  ▼
evaluate    hard: crash rate ≤ 10%, every DENY has a reason code
            soft: 0.4·accuracy + 0.2·simplicity + 0.2·robustness + 0.2·coverage
  ▼
evolve      top-K (or behaviorally diverse set) seed the next generation; every
            generation is written to disk
  ▼
best_underwriter.py                                                            [human review]
```

Also available: cross-model evolution (island-model GA pooling candidates from several LLMs)
and an active-learning loop (label the cases where programs disagree most, then re-evolve).

## Setup

```bash
pip install -e ".[dev]"          # core + pytest; ".[all]" adds anthropic, openai, analysis, baselines
cp .env.example .env             # fill in the API key(s) you use
```

Python ≥ 3.10. Providers (`--provider`): `anthropic`, `openai`, `gemini`, `deepseek`,
`together`, `openrouter` (the last four via the OpenAI-compatible client).

## Run the HELOC example

```bash
# 1. Data — not redistributed; see datasets/heloc/GET_DATA.md
python datasets/heloc/fetch_heloc.py --input path/to/heloc_dataset_v1.csv
#    (or, with `pip install datasets`: python datasets/heloc/fetch_heloc.py   ← Hugging Face mstz/heloc)
#    (no data: python datasets/heloc/generate_synthetic.py  → data/heloc_synthetic.csv)

# 2. Targets, then review/correct the JSON
solon targets --dataset datasets/heloc/dataset.yaml \
  --cases data/heloc_100_cases.csv --provider anthropic --output outputs

# 3. Evolve programs against the reviewed targets
solon evolve --dataset datasets/heloc/dataset.yaml \
  --cases data/heloc_100_cases.csv --full-dataset data/heloc_real.csv \
  --targets outputs/targets_<model>.json --generations 5
```

The shipped targets in `examples/heloc/targets_claude-opus-4-6.json` can be reused in step 3
as-is; their dataset rows are not shipped and are re-joined from `--cases` by `case_index`.

`solon -h` lists all subcommands: `sample`, `cluster`, `targets`, `eval-targets`,
`synthesize`, `eval-program`, `evolve`, `cross-evolve`, `active-learn`, and `analyze`
(`compare`, `similarity`, `decision-boundary`) for looking at finished evolve runs.

## Layout

```
src/solon/   config/ llm/ pipeline/ (targets, synthesis, evaluator, evolve)
                    prompts/ sampling/ datasets/ analysis/  cli.py
datasets/heloc/     dataset.yaml, fetch_heloc.py, generate_synthetic.py, results.md
examples/heloc/     one targets file (reasoning numbers redacted), the best program and
                    history of each evolve run, the cross-model winner's eval, cluster
                    profiles (derived; no dataset rows)
refs/               INDEX.md reading list + download_papers.sh (papers not included)
tests/              pytest suite (fixtures are hand-made, not dataset rows)
```

A new dataset is one YAML file (schema: `src/solon/config/schema.py`; example:
`datasets/heloc/dataset.yaml`).

## Tests

```bash
pytest
```

## Caveats

- Synthesized programs run in-process. `validate_program` blocks a denylist of dangerous calls
  and target leakage, but it is **not a sandbox** — treat generated code as untrusted.
- Targets come from an LLM; if they are wrong, evolution converges on wrong answers. Review them.
- Reason codes are FCRA-style labels produced by the model; nothing here is a compliance review.

## Background

Builds on FunSearch (Romera-Paredes et al., 2024), CodeEvolve, and the Gödel-machine line of
work (Schmidhuber 2003; Huxley-Gödel Machine): the LLM is treated as a fixed oracle and the
programs around it are what evolve. See `refs/INDEX.md`.

## Data

The FICO HELOC dataset comes from the FICO Explainable Machine Learning Challenge and is
subject to FICO's terms; it is not included, and `fetch_heloc.py` rebuilds it from your own copy
(FICO's download, Kaggle, or Hugging Face `mstz/heloc`). FICO's terms for redistributing derived
text are unclear, so every number in the shipped targets' `reasoning` text is replaced with `[n]`.
Example outputs in `examples/heloc/` were produced by Claude Opus 4.6, GPT-5.2, Gemini 3 Pro and
Gemini 3 Flash.

## License

MIT — see [LICENSE](LICENSE). © 2026 Ramakrishna Kompella.
