# Lending Club Dataset Design

## Overview

Add Lending Club peer-to-peer lending data as a second dataset for the solon pipeline. Uses the raw accepted loans file (2.26M rows, 151 columns) filtered to resolved outcomes. Programs make approve/deny decisions on unsecured personal loan applications using 58 features including free-text job titles.

Key difference from HELOC: includes `emp_title` as raw text, testing whether evolved programs can discover occupational risk patterns via string matching.

## Data Source

- **Raw file**: `data/lendingclub_raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`
- **Rows**: 2,260,701 total; ~1,345,310 resolved (Fully Paid + Charged Off)
- **Target**: `loan_status` — Fully Paid (positive, ~80%) vs Charged Off (negative, ~20%)
- **Filter**: Drop `Current`, `Late (16-30 days)`, `Late (31-120 days)`, `In Grace Period`, `Default`

## Excluded Fields

| Field | Reason |
|-------|--------|
| `grade`, `sub_grade`, `int_rate` | LC-assigned risk tiers — encodes the same credit features, creates shortcut |
| `installment` | Derived from loan_amnt + term + int_rate |
| `desc` | 91% empty (LC stopped collecting after 2013) |
| `title` | Redundant with `purpose` (only 15 values, mirrors the enum) |
| `earliest_cr_line` | Date string requiring parsing; `mo_sin_old_rev_tl_op` captures same info |
| `application_type` | 99.9% "Individual" — effectively constant |
| All post-issuance fields | `last_pymnt_d`, `total_pymnt`, `recoveries`, etc. — not available at origination |
| All hardship/settlement fields | Post-issuance outcomes |
| All `sec_app_*` fields | Joint application fields, >99% empty |
| `member_id`, `id`, `url`, `policy_code` | Identifiers, not features |

## Features (58 total)

### Loan Characteristics (2)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `loan_amnt` | numeric | 0% | 1,000-35,000 | Median: 12,425. P25: 8,000, P75: 20,000 |
| `term` | categorical | 0% | 36 months, 60 months | 76% are 36-month |

### Borrower Profile (7)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `emp_title` | **string (free text)** | 5.9% | 411k unique values | Top: teacher, manager, owner, registered nurse, driver. Signal: attorney 11.6% default vs truck driver 29.5% |
| `emp_length` | categorical | 5.9% | < 1 year to 10+ years | 33% are "10+ years" |
| `home_ownership` | categorical | 0% | MORTGAGE, RENT, OWN | MORTGAGE 49%, RENT 40%, OWN 11% |
| `annual_inc` | numeric | 0% | 0-9,500,000 | Median: 65,000. P25: 45,000, P75: 90,500 |
| `verification_status` | categorical | 0% | Not Verified, Source Verified, Verified | Source Verified 43%, Verified 30%, Not Verified 27% |
| `purpose` | categorical | 0% | 14 categories | debt_consolidation 58%, credit_card 24%, home_improvement 6% |
| `dti` | numeric | 0% | 0-999 | Median: 18. P25: 12, P75: 25 |

### FICO Scores (2)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `fico_range_low` | numeric | 0% | 660-845 | Median: 685. P25: 670, P75: 710. LC has 660 floor. |
| `fico_range_high` | numeric | 0% | 664-850 | Median: 689. P25: 674, P75: 714 |

### Credit History Depth (6)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `open_acc` | numeric | 0% | 1-82 | Median: 11. P25: 8, P75: 15 |
| `total_acc` | numeric | 0% | 4-169 | Median: 24. P25: 17, P75: 32 |
| `num_sats` | numeric | 0% | 0-82 | Median: 11. P25: 8, P75: 15 |
| `mo_sin_old_il_acct` | numeric | 3% | 0-724 | Median: 129 (10.8 years). P25: 99, P75: 153 |
| `mo_sin_old_rev_tl_op` | numeric | 0% | 3-791 | Median: 168 (14 years). P25: 118, P75: 235 |
| `mort_acc` | numeric | 0% | 0-47 | Median: 1. P75: 3, P95: 5 |

### Delinquency History (11)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `delinq_2yrs` | numeric | 0% | 0-39 | Median: 0. P95: 2. Most borrowers have none. |
| `mths_since_last_delinq` | numeric | **48%** | 0-171 | Median: 31. **Missing = no delinquency on record.** |
| `mths_since_last_record` | numeric | **82%** | 0-120 | Median: 67. **Missing = no public record.** |
| `mths_since_last_major_derog` | numeric | **70%** | 0-171 | Median: 44. **Missing = no major derogatory.** |
| `acc_now_delinq` | numeric | 0% | 0-6 | Nearly all 0. |
| `num_accts_ever_120_pd` | numeric | 0% | 0-39 | Median: 0. P95: 3 |
| `num_tl_30dpd` | numeric | 0% | 0-4 | Nearly all 0. |
| `num_tl_90g_dpd_24m` | numeric | 0% | 0-39 | Median: 0. P95: 1 |
| `pct_tl_nvr_dlq` | numeric | 0% | 11-100 | Median: 97. P25: 91 |
| `mths_since_recent_bc_dlq` | numeric | **74%** | 0-189 | Median: 38. **Missing = no bankcard delinquency.** |
| `mths_since_recent_revol_delinq` | numeric | **64%** | 0-171 | Median: 33. **Missing = no revolving delinquency.** |

### Inquiries & Recent Activity (7)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `inq_last_6mths` | numeric | 0% | 0-5 | Median: 0. P75: 1 |
| `mths_since_recent_inq` | numeric | 10% | 0-25 | Median: 5. |
| `acc_open_past_24mths` | numeric | 0% | 0-64 | Median: 4. P75: 6 |
| `mo_sin_rcnt_tl` | numeric | 0% | 0-263 | Median: 5. P75: 10 |
| `mo_sin_rcnt_rev_tl_op` | numeric | 0% | 0-324 | Median: 8. P75: 16 |
| `num_tl_op_past_12m` | numeric | 0% | 0-30 | Median: 2. P75: 3 |
| `mths_since_recent_bc` | numeric | 0.9% | 0-610 | Median: 13. P75: 29 |

### Revolving Credit (7)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `revol_bal` | numeric | 0% | 0-2,904,836 | Median: 11,416. P75: 20,482 |
| `revol_util` | numeric | 0% | 0-193 | Median: 53%. P25: 35, P75: 71 |
| `total_rev_hi_lim` | numeric | 0% | 0-9,999,999 | Median: 24,100. P75: 41,000 |
| `num_rev_accts` | numeric | 0% | 2-112 | Median: 13. P75: 19 |
| `num_actv_rev_tl` | numeric | 0% | 0-52 | Median: 5. P75: 7 |
| `num_rev_tl_bal_gt_0` | numeric | 0% | 0-45 | Median: 5. P75: 7 |
| `num_op_rev_tl` | numeric | 0% | 0-67 | Median: 7. P75: 11 |

### Bankcard Specific (6)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `bc_open_to_buy` | numeric | 0.9% | 0-332,178 | Median: 4,468. P75: 11,706 |
| `bc_util` | numeric | 1% | 0-235 | Median: 64%. P75: 86 |
| `num_actv_bc_tl` | numeric | 0% | 0-32 | Median: 3. P75: 5 |
| `num_bc_sats` | numeric | 0% | 0-57 | Median: 4. P75: 6 |
| `num_bc_tl` | numeric | 0% | 0-61 | Median: 7. P75: 10 |
| `percent_bc_gt_75` | numeric | 1% | 0-100 | Median: 50. P75: 75 |

### Balances & Limits (7)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `tot_cur_bal` | numeric | 0% | 0-4,447,397 | Median: 76,300. P75: 203,762 |
| `avg_cur_bal` | numeric | 0% | 0-555,925 | Median: 7,034. P75: 17,728 |
| `total_bal_ex_mort` | numeric | 0% | 0-2,921,551 | Median: 38,808. P75: 64,579 |
| `tot_hi_cred_lim` | numeric | 0% | 2,500-9,999,999 | Median: 107,647. P75: 245,718 |
| `total_bc_limit` | numeric | 0% | 0-344,200 | Median: 14,900. P75: 28,000 |
| `num_il_tl` | numeric | 0% | 0-132 | Median: 7. P75: 11 |
| `total_il_high_credit_limit` | numeric | 0% | 0-2,101,913 | Median: 32,693. P75: 57,642 |

### Public Records & Collections (6)

| Feature | Type | Missing | Range | Distribution |
|---------|------|---------|-------|-------------|
| `pub_rec` | numeric | 0% | 0-86 | Nearly all 0. P95: 1 |
| `pub_rec_bankruptcies` | numeric | 0% | 0-9 | Nearly all 0. P95: 1 |
| `tax_liens` | numeric | 0% | 0-85 | Nearly all 0. |
| `collections_12_mths_ex_med` | numeric | 0% | 0-10 | Nearly all 0. |
| `tot_coll_amt` | numeric | 0% | 0-380,757 | Median: 0. P95: 860 |
| `chargeoff_within_12_mths` | numeric | 0% | 0-10 | Nearly all 0. |

## Summary Features

Features shown in misclassification reports:

```yaml
summary_features:
  - fico_range_low
  - dti
  - annual_inc
  - revol_util
  - delinq_2yrs
  - inq_last_6mths
  - pct_tl_nvr_dlq
  - emp_title
```

## Missing Data Strategy

No sentinel values (unlike HELOC's -7/-8/-9). Missing values are empty/NaN, handled by the `get()` helper returning `None`.

For "months since" features with high missing rates (48-82%), missing means "no event on record" — this must be documented in feature descriptions so LLMs understand the semantics.

## Regulatory Context

- **FCRA**: Same adverse action reason code requirements as HELOC
- **ECOA**: No demographic features in dataset, compliance is inherent
- **TILA (Truth in Lending Act)**: Requires clear disclosure of loan terms; relevant for unsecured consumer lending
- **State usury laws**: LC operated within specific rate caps per state

## Data Observations

- Default rate is ~20% (vs HELOC's 52%) — moderately imbalanced toward good outcomes
- FICO range is narrow (660-850) due to LC's 660 minimum floor
- 58% of loans are for debt consolidation — relatively homogeneous purpose mix
- `emp_title` has real predictive signal (attorney 11.6% default vs truck driver 29.5%) but is extremely noisy (411k unique values, many typos)
- High-missing "months since" features (48-82%) encode informative missingness — "no event on record" is a positive signal
- Population skews toward borrowers who may not qualify for traditional bank loans

## Decision Framework

For each application, assess:

1. **Creditworthiness** — FICO score, delinquency history, public records
2. **Capacity** — annual income, DTI ratio, existing debt burden
3. **Leverage** — revolving utilization, bankcard utilization, total balances relative to limits
4. **Recent behavior** — inquiries, new accounts, recency of delinquency
5. **Stability** — employment length, home ownership, credit history depth
6. **Loan characteristics** — amount relative to income, term choice
7. **Occupational signal** — job title patterns (if program chooses to use emp_title)

## Evaluation Config

```yaml
evaluation:
  prohibited_features: []
  require_reason_codes: true
  max_crash_rate: 0.1
  weights:
    accuracy: 0.4
    simplicity: 0.2
    robustness: 0.2
    coverage: 0.2
  simplicity_thresholds:
    ideal_max: 15
    good_max: 30
    acceptable_max: 50
```

Simplicity thresholds bumped from HELOC (10/20/40) since 58 features naturally leads to more conditions.

## Synthesizer Config

```yaml
synthesizer:
  function_name: underwrite
  return_fields: [decision, reason_codes, score]

defaults:
  sample_size: 100
  n_clusters: 20
  holdout_size: 500
  batch_size: 10
  n_candidates: 5
  n_generations: 10
  top_k: 3
```

## Preprocessing Steps

1. Read raw CSV: `data/lendingclub_raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`
2. Filter to `loan_status` in (`Fully Paid`, `Charged Off`)
3. Keep only the 58 feature columns + `loan_status`
4. Rename `loan_status` values: keep as-is (strings "Fully Paid" / "Charged Off")
5. Save as `data/lendingclub.csv`
6. Add `data/lendingclub_raw/` to `.gitignore`
7. Create `datasets/lendingclub/dataset.yaml` with full config

## No Code Changes Required

The pipeline is fully config-driven. Creating `datasets/lendingclub/dataset.yaml` and the preprocessed CSV is sufficient. The `get()` helper in synthesized programs already handles None for missing values. The only consideration is that `emp_title` is a string type — the `get()` helper should return it as-is (not try to parse as numeric), which it already does since it just returns `application.get(key)`.
