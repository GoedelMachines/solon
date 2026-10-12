# Lending Club Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Lending Club as a second dataset for solon, including preprocessing, config YAML, template fixes for mixed-type features, and tests.

**Architecture:** Config-driven — create `datasets/lendingclub/dataset.yaml` following the HELOC pattern. Fix templates to handle empty `special_values` and mixed feature types. Preprocess raw data into `data/lendingclub.csv`. No schema changes needed.

**Tech Stack:** Python, pandas, PyYAML, Jinja2, pytest

---

### Task 1: Add lendingclub_raw to .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add the raw data directory to gitignore**

Add under the existing `# Data files` section:

```
# Raw data (too large for git)
data/lendingclub_raw/
```

**Step 2: Verify**

Run: `git status`
Expected: `data/lendingclub_raw/` no longer shows as untracked

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore lendingclub raw data (4GB)"
```

---

### Task 2: Fix templates for empty special_values and mixed feature types

The current templates break when `special_values.values` is empty (renders `(,)` — Python syntax error) and assume all features share a single type. LC has numeric, categorical, and string features with no sentinel values.

**Files:**
- Modify: `src/solon/prompts/templates.py:10-165`
- Test: `tests/test_prompts.py`

**Step 1: Write failing tests**

Add to `tests/test_prompts.py`:

```python
class TestEmptySpecialValues:
    """Templates must work when special_values.values is empty (e.g., Lending Club)."""

    def test_skeleton_no_special_values(self):
        """get() helper must be valid Python when no special values exist."""
        from solon.prompts.templates import render_program_skeleton
        from solon.config.schema import DatasetConfig, SpecialValueSpec
        cfg = DatasetConfig(
            name="test",
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
        )
        skeleton = render_program_skeleton(cfg)
        # Must be valid Python
        compile(skeleton, "<skeleton>", "exec")
        # Must NOT contain empty tuple syntax (,)
        assert "(,)" not in skeleton

    def test_synthesis_prompt_no_special_values(self):
        """Synthesis prompt must not mention 'Special values []'."""
        from solon.prompts.templates import render_synthesis_system_prompt
        from solon.config.schema import (
            DatasetConfig, SpecialValueSpec, FeatureSpec, SynthesizerConfig
        )
        cfg = DatasetConfig(
            name="test",
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
            features=[FeatureSpec(name="loan_amnt", description="Loan amount", type="numeric")],
            synthesizer=SynthesizerConfig(),
        )
        prompt = render_synthesis_system_prompt(cfg)
        assert "Special values []" not in prompt

    def test_target_prompt_mixed_types(self):
        """Target prompt must handle mixed feature types without 'all X-valued'."""
        from solon.prompts.templates import render_target_system_prompt
        from solon.config.schema import (
            DatasetConfig, SpecialValueSpec, FeatureSpec, DecisionSpec
        )
        cfg = DatasetConfig(
            name="test",
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
            features=[
                FeatureSpec(name="loan_amnt", description="Loan amount", type="numeric"),
                FeatureSpec(name="emp_title", description="Job title", type="string"),
            ],
            decisions=DecisionSpec(),
        )
        prompt = render_target_system_prompt(cfg)
        # Should not say "All features are numeric-valued" when types are mixed
        assert "All features are numeric-valued" not in prompt
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py::TestEmptySpecialValues -v`
Expected: At least 2 of 3 FAIL (skeleton syntax error, "Special values []" present)

**Step 3: Fix SKELETON_TEMPLATE**

In `src/solon/prompts/templates.py`, change the `get()` helper in `SKELETON_TEMPLATE` from:

```python
    def get(feature, default=None):
        val = application.get(feature)
        if val in ({{ special_values.values | join(', ') }},) or val is None:
            return default
        return val
```

to:

```python
    def get(feature, default=None):
        val = application.get(feature)
{% if special_values.values %}
        if val in ({{ special_values.values | join(', ') }},) or val is None:
{% else %}
        if val is None or (isinstance(val, float) and val != val):
{% endif %}
            return default
        return val
```

The `val != val` check catches NaN floats (from pandas CSV loading).

**Step 4: Fix SYNTHESIS_SYSTEM_TEMPLATE**

Change the special values line from:

```
Special values {{ special_values.values }} mean missing/unavailable data. Use the get() helper.
```

to:

```
{% if special_values.values %}
Special values {{ special_values.values }} mean missing/unavailable data. Use the get() helper.
{% else %}
Missing values appear as None or NaN. Use the get() helper to safely access features with a default.
{% endif %}
```

**Step 5: Fix TARGET_SYSTEM_TEMPLATE feature type line**

Change:

```
All features are {{ feature_type }}-valued. The distributions below are from the
actual applicant pool.
```

to:

```
{% if mixed_types %}
Features are a mix of numeric, categorical, and string types (noted in each definition).
{% else %}
All features are {{ feature_type }}-valued.
{% endif %}
The distributions below are from the actual applicant pool.
```

And in `render_target_system_prompt()`, compute `mixed_types`:

```python
    types = set(f.type for f in config.features if f.type)
    mixed_types = len(types) > 1
```

Pass `mixed_types=mixed_types` to the template render call.

**Step 6: Fix the special values block in TARGET_SYSTEM_TEMPLATE**

Wrap the special values enumeration in a conditional:

```
{% if special_values.values %}
{% for sv in special_values.values -%}
Special value {{ sv }} = "{{ special_values.meanings.get(sv, 'special') }}"{{ ", " if not loop.last else "." }}
{%- endfor %}
 Treat these as MISSING DATA, not as numeric values.
{% else %}
Missing or unavailable values appear as blank/empty. When a value is missing, it may carry meaning (e.g., "months since last delinquency" being missing means no delinquency on record).
{% endif %}
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: All pass, including the new `TestEmptySpecialValues` tests and all existing tests.

**Step 8: Commit**

```bash
git add src/solon/prompts/templates.py tests/test_prompts.py
git commit -m "fix: handle empty special_values and mixed feature types in templates"
```

---

### Task 3: Write the preprocessing script

**Files:**
- Create: `datasets/lendingclub/preprocess.py`

**Step 1: Write the script**

```python
"""
Preprocess raw Lending Club data into pipeline-ready CSV.

Reads the raw accepted loans file, filters to resolved outcomes
(Fully Paid / Charged Off), selects the 58 origination features,
and saves to data/lendingclub.csv.

Usage: python datasets/lendingclub/preprocess.py
"""

import os
import pandas as pd

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "lendingclub_raw",
    "accepted_2007_to_2018q4.csv", "accepted_2007_to_2018Q4.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "lendingclub.csv"
)

KEEP_COLUMNS = [
    # Loan characteristics
    "loan_amnt", "term",
    # Borrower profile
    "emp_title", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti",
    # FICO
    "fico_range_low", "fico_range_high",
    # Credit history depth
    "open_acc", "total_acc", "num_sats", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mort_acc",
    # Delinquency
    "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record",
    "mths_since_last_major_derog", "acc_now_delinq", "num_accts_ever_120_pd",
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "pct_tl_nvr_dlq",
    "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq",
    # Inquiries & recent activity
    "inq_last_6mths", "mths_since_recent_inq", "acc_open_past_24mths",
    "mo_sin_rcnt_tl", "mo_sin_rcnt_rev_tl_op", "num_tl_op_past_12m",
    "mths_since_recent_bc",
    # Revolving credit
    "revol_bal", "revol_util", "total_rev_hi_lim", "num_rev_accts",
    "num_actv_rev_tl", "num_rev_tl_bal_gt_0", "num_op_rev_tl",
    # Bankcard
    "bc_open_to_buy", "bc_util", "num_actv_bc_tl", "num_bc_sats",
    "num_bc_tl", "percent_bc_gt_75",
    # Balances & limits
    "tot_cur_bal", "avg_cur_bal", "total_bal_ex_mort", "tot_hi_cred_lim",
    "total_bc_limit", "num_il_tl", "total_il_high_credit_limit",
    # Public records & collections
    "pub_rec", "pub_rec_bankruptcies", "tax_liens",
    "collections_12_mths_ex_med", "tot_coll_amt", "chargeoff_within_12_mths",
    # Target
    "loan_status",
]

RESOLVED_STATUSES = ["Fully Paid", "Charged Off"]


def main():
    print(f"Reading {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, usecols=KEEP_COLUMNS, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    df = df[df["loan_status"].isin(RESOLVED_STATUSES)]
    print(f"  After filtering to resolved: {len(df):,}")

    # Strip whitespace from term values
    df["term"] = df["term"].str.strip()

    # Report
    default_rate = (df["loan_status"] == "Charged Off").mean()
    print(f"  Default rate: {default_rate:.1%}")
    print(f"  Features: {len(KEEP_COLUMNS) - 1}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

Run: `python datasets/lendingclub/preprocess.py`
Expected output:
```
Reading data/lendingclub_raw/...
  Raw rows: 2,260,701
  After filtering to resolved: ~1,345,310
  Default rate: ~20.0%
  Features: 58
  Saved to data/lendingclub.csv
```

**Step 3: Verify output**

Run: `head -2 data/lendingclub.csv | cut -c1-200` and `wc -l data/lendingclub.csv`
Expected: Header with 59 columns (58 features + loan_status), ~1.35M rows.

**Step 4: Commit**

```bash
git add datasets/lendingclub/preprocess.py
git commit -m "feat: add Lending Club preprocessing script"
```

---

### Task 4: Create datasets/lendingclub/dataset.yaml

This is the core config file. 58 features with full descriptions, distributions, and missing rates. All content drawn from the design doc.

**Files:**
- Create: `datasets/lendingclub/dataset.yaml`

**Step 1: Write the YAML**

Create `datasets/lendingclub/dataset.yaml` with all 58 features, target/decision config, regulatory context, data observations, decision framework, evaluation config, synthesizer config, and defaults. Reference the design doc (`docs/plans/2026-02-18-lendingclub-dataset-design.md`) for exact values.

Key structural points:
- `target.positive_label: "Fully Paid"` and `target.negative_label: "Charged Off"` (strings, not ints)
- `special_values.values: []` (empty — no sentinel values)
- Feature types: most are `numeric`, `term`/`emp_length`/`home_ownership`/`verification_status`/`purpose` are `categorical`, `emp_title` is `string`
- High-missing features (mths_since_last_delinq etc.) must note "Missing = no X on record" in description
- `simplicity_thresholds`: ideal_max: 15, good_max: 30, acceptable_max: 50

**Step 2: Validate YAML loads**

Run: `python -c "from solon.config import load_dataset_config; c = load_dataset_config('datasets/lendingclub/dataset.yaml'); print(f'{c.name}: {len(c.features)} features')"`
Expected: `lendingclub: 58 features`

**Step 3: Commit**

```bash
git add datasets/lendingclub/dataset.yaml
git commit -m "feat: add Lending Club dataset config (58 features)"
```

---

### Task 5: Add test fixtures for Lending Club

**Files:**
- Create: `tests/fixtures/lendingclub_dataset.yaml` (mini config, 3-4 features, mixed types)
- Create: `tests/fixtures/tiny_lendingclub.csv` (5 rows with mixed types including emp_title)
- Modify: `tests/conftest.py` (add fixtures)

**Step 1: Create mini YAML fixture**

```yaml
name: lendingclub_mini
description: Minimal Lending Club config for testing

target:
  column: loan_status
  positive_label: "Fully Paid"
  negative_label: "Charged Off"
  positive_meaning: "borrower repaid"
  negative_meaning: "borrower defaulted"

decisions:
  approve: APPROVE
  deny: DENY
  approve_maps_to: positive
  deny_maps_to: negative

special_values:
  values: []
  meanings: {}
  treatment: missing

features:
  - name: fico_range_low
    description: "Lower bound of FICO range at origination"
    type: numeric
    range: "660-845"
    direction: higher_better
  - name: dti
    description: "Debt-to-income ratio"
    type: numeric
    range: "0-999"
    direction: lower_better
  - name: emp_title
    description: "Free-text job title"
    type: string
  - name: purpose
    description: "Loan purpose category"
    type: categorical

summary_features:
  - fico_range_low
  - dti

regulatory_context: "Test regulatory context."
data_observations: "Test observations."
decision_framework: "Test framework."
product_context: "Test product."

evaluation:
  prohibited_features: []
  require_reason_codes: true
  max_crash_rate: 0.1
  weights:
    accuracy: 0.4
    simplicity: 0.2
    robustness: 0.2
    coverage: 0.2

defaults:
  sample_size: 10
  n_clusters: 3
  holdout_size: 20
  batch_size: 5
  n_candidates: 2
  n_generations: 2
  top_k: 1
```

**Step 2: Create tiny CSV fixture**

```csv
fico_range_low,dti,emp_title,purpose,annual_inc,revol_util,loan_amnt,loan_status
700,15.5,teacher,debt_consolidation,65000,45.2,12000,Fully Paid
665,28.3,truck driver,credit_card,42000,78.1,25000,Charged Off
720,10.2,software engineer,home_improvement,95000,22.0,8000,Fully Paid
680,35.1,,small_business,38000,92.5,15000,Charged Off
690,18.7,registered nurse,debt_consolidation,72000,55.3,20000,Fully Paid
```

**Step 3: Add fixtures to conftest.py**

Add to `tests/conftest.py`:

```python
@pytest.fixture
def lendingclub_yaml_path():
    return os.path.join(FIXTURES_DIR, "lendingclub_dataset.yaml")

@pytest.fixture
def full_lendingclub_yaml_path():
    return os.path.join(
        os.path.dirname(__file__), "..", "datasets", "lendingclub", "dataset.yaml"
    )

@pytest.fixture
def tiny_lendingclub_dataset():
    return pd.read_csv(os.path.join(FIXTURES_DIR, "tiny_lendingclub.csv"))
```

**Step 4: Commit**

```bash
git add tests/fixtures/lendingclub_dataset.yaml tests/fixtures/tiny_lendingclub.csv tests/conftest.py
git commit -m "test: add Lending Club test fixtures"
```

---

### Task 6: Add tests for Lending Club config loading

**Files:**
- Modify: `tests/test_config.py`

**Step 1: Write tests**

Add to `tests/test_config.py`:

```python
class TestLoadLendingClubConfig:
    def test_loads_mini_config(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        assert isinstance(cfg, DatasetConfig)
        assert cfg.name == "lendingclub_mini"

    def test_target_fields(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        assert cfg.target.column == "loan_status"
        assert cfg.target.positive_label == "Fully Paid"
        assert cfg.target.negative_label == "Charged Off"

    def test_empty_special_values(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        assert cfg.special_values.values == []
        assert cfg.special_values.meanings == {}

    def test_mixed_feature_types(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        types = set(f.type for f in cfg.features)
        assert "numeric" in types
        assert "string" in types
        assert "categorical" in types

    def test_string_feature_present(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        emp = [f for f in cfg.features if f.name == "emp_title"][0]
        assert emp.type == "string"


class TestLoadFullLendingClubConfig:
    def test_loads_full_config(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        assert cfg.name == "lendingclub"
        assert len(cfg.features) == 58

    def test_all_features_populated(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        for f in cfg.features:
            assert f.name, f"Feature missing name: {f}"
            assert f.description, f"Feature {f.name} missing description"

    def test_summary_features_are_valid(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        names = cfg.feature_names
        for sf in cfg.summary_features:
            assert sf in names, f"Summary feature {sf} not in feature list"

    def test_string_labels(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        assert isinstance(cfg.target.positive_label, str)
        assert isinstance(cfg.target.negative_label, str)

    def test_bumped_simplicity_thresholds(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        assert cfg.evaluation.simplicity_thresholds["ideal_max"] == 15
        assert cfg.evaluation.simplicity_thresholds["good_max"] == 30
        assert cfg.evaluation.simplicity_thresholds["acceptable_max"] == 50
```

**Step 2: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All pass (existing HELOC tests + new LC tests)

**Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test: add Lending Club config loading tests"
```

---

### Task 7: Run preprocessing and smoke test

**Step 1: Run preprocessing**

Run: `python datasets/lendingclub/preprocess.py`
Expected: Saves `data/lendingclub.csv` (~1.35M rows)

**Step 2: Smoke test — sample subcommand**

Run: `solon sample --dataset datasets/lendingclub/dataset.yaml --cases data/lendingclub.csv --size 10`
Expected: Outputs 10 sampled rows without errors. Verifies the full config → sampling pipeline works.

**Step 3: Verify template rendering**

Run:
```python
python -c "
from solon.config import load_dataset_config
from solon.prompts.templates import render_target_system_prompt, render_synthesis_system_prompt, render_program_skeleton

cfg = load_dataset_config('datasets/lendingclub/dataset.yaml')
p1 = render_target_system_prompt(cfg)
p2 = render_synthesis_system_prompt(cfg)
p3 = render_program_skeleton(cfg)
compile(p3, '<skeleton>', 'exec')
print(f'Target prompt: {len(p1)} chars')
print(f'Synthesis prompt: {len(p2)} chars')
print(f'Skeleton compiles: OK')
print('All templates render successfully')
"
```

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new)

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete Lending Club dataset integration"
```
