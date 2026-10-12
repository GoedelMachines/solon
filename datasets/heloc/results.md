# HELOC Dataset Results

## Dataset

FICO Home Equity Line of Credit (HELOC) dataset. 10,459 real credit applications, 23 integer-valued credit bureau features, binary outcome (52% default rate). No demographic features.

Source: FICO Explainable Machine Learning Challenge.

## Method

Oracle-guided evolutionary program synthesis:
1. An LLM "oracle" labels a 100-case stratified sample with APPROVE/DENY decisions + reason codes
2. LLM synthesizers generate deterministic Python `underwrite()` functions that replicate the oracle's decisions
3. Candidates are evaluated on accuracy, simplicity, robustness, and coverage
4. Best programs seed the next generation (misclassification feedback provided)
5. Final programs are evaluated against the full 10,459-case dataset

All synthesized programs are pure Python with no ML dependencies — rule-based decision trees using `if/elif/else` logic with explicit thresholds.

## Single-Model Evolve Results (10 generations each)

| Model | Best Acc (full) | Best F1 (full) | Internal Score | Notes |
|---|---|---|---|---|
| Gemini 3 Flash | **0.704** | **0.699** | 0.812 | Fast convergence, best early F1 |
| Gemini 3 Pro | 0.703 | 0.696 | 0.819 | Similar ceiling to Flash |
| Claude Opus 4.6 | 0.706 | 0.701 | 0.772 | Highest single-model F1 |
| GPT-5.2 | 0.689 | 0.632 | 0.784 | Slowest to converge, lower ceiling |

All models converge to ~70% accuracy on the full dataset despite 80-90%+ accuracy on the 100-case training sample. The gap between internal score and full-dataset accuracy reflects overfitting to the small training set.

## Cross-Model Evolve Results

4 models (Flash, Pro, GPT-5.2, Opus) x 3 candidates per model x 10 generations.

| Metric | Value |
|---|---|
| Best accuracy (full dataset) | **0.707** |
| Best F1 (full dataset) | **0.699** |
| Best internal score | 0.830 |
| Best source (internal) | Gemini Pro (gen 9) |

### Convergence Timeline (internal scores)

| Gen | Flash | Pro | GPT-5.2 | Opus |
|---|---|---|---|---|
| 1 | 0.812 | 0.795 | 0.670 | 0.743 |
| 3 | 0.824 | 0.799 | 0.770 | 0.770 |
| 5 | 0.820 | 0.823 | 0.790 | 0.793 |
| 7 | 0.823 | 0.825 | 0.816 | 0.814 |
| 10 | 0.825 | 0.825 | 0.818 | 0.823 |

Cross-pollination works: GPT-5.2 improved from 0.670 to 0.818 by learning from Flash/Pro/Opus seeds. By gen 6, all models converge to producing functionally identical programs (0.707 acc, 0.685 F1, 45.4% approval rate on the full dataset).

## Convergence Analysis

### Behavioral Similarity

Across all 184 single-model candidates + 120 cross-evolve candidates:
- 7 behavioral clusters identified, with one dominant cluster containing 106/184 programs
- Flash vs Pro: 89.3% average pairwise agreement
- GPT-5.2 outlier at 69.6% agreement (early generations)
- 3,794 cases (36%) fall in high-disagreement regions between programs

### Code Structure Convergence

By gen 5 of cross-evolve, all models converge to ~15 core features:
- ExternalRiskEstimate (primary signal — all programs use tiered thresholds)
- NetFractionRevolvingBurden
- PercentTradesNeverDelq
- NumSatisfactoryTrades
- MSinceOldestTradeOpen
- NumInqLast6M
- MSinceMostRecentDelq
- NumTrades60Ever2DerogPubRec
- PercentTradesWBalance
- NumBank2NatlTradesWHighUtilization

### Decision Landscape

UMAP projection of 10,459 cases with program consensus overlaid:
- **mostly_deny**: 5,235 cases, 71.8% bad rate
- **lean_deny**: 502 cases, 58.2% bad rate
- **lean_approve**: 708 cases, 48.0% bad rate
- **mostly_approve**: 4,014 cases, 26.6% bad rate

The 1,210 "lean" cases (11.6% of data) are the contested zone where programs disagree. These borderline cases have a bad rate close to the population average (50.6%), confirming they are genuinely ambiguous.

## Best Synthesized Program

The best program (`examples/heloc/cross_evolve/best_underwriter.py`, 160 lines) implements:
- Tiered decision logic based on ExternalRiskEstimate (ERE):
  - ERE < 64: automatic deny
  - ERE 64-69: deny unless deep clean file (100% never delq, 20+ sat trades, 100+ months oldest)
  - ERE 70-74: deny for recent delinquency or high revolving burden
  - ERE 75-79: deny for thin files with many inquiries
  - ERE 80+: deny only for extremely thin files
- Global safety net: deny if <70% trades never delinquent
- FCRA-style adverse-action reason codes for both approvals and denials (not a compliance review)

## Key Finding

The ~70% accuracy ceiling appears to be a fundamental limit of the 100-case training approach, not a model capability limit. All 4 models independently converge to the same decision boundary regardless of code complexity (122-445 lines) or feature count (0-23). The programs agree on ~90% of cases and disagree primarily on borderline applicants with near-50% default rates — cases where the correct answer is genuinely uncertain given the available features.

Potential paths to break through:
1. **Larger training samples** — 500-1000 cases instead of 100
2. **Clade-based diversity selection** — maintain behavioral diversity across generations (in progress)
3. **Ensemble methods** — combine decisions from diverse programs rather than selecting one best
