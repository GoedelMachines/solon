"""
Preprocess raw Lending Club loan data for the solon pipeline.

Reads the full accepted loans file (2007-2018 Q4), filters to terminal
loan statuses (Fully Paid / Charged Off), selects 58 feature columns
plus the target, cleans whitespace artifacts, and writes a pipeline-ready
CSV to data/lendingclub.csv.

Raw data source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
Expected input:  data/lendingclub_raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
Expected output: data/lendingclub.csv
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script so it works from any working directory)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")

RAW_PATH = os.path.join(
    _PROJECT_ROOT,
    "data",
    "lendingclub_raw",
    "accepted_2007_to_2018q4.csv",
    "accepted_2007_to_2018Q4.csv",
)
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "lendingclub.csv")

# ---------------------------------------------------------------------------
# Target column and terminal statuses
# ---------------------------------------------------------------------------
TARGET = "loan_status"
KEEP_STATUSES = {"Fully Paid", "Charged Off"}

# ---------------------------------------------------------------------------
# 58 feature columns grouped by domain
# ---------------------------------------------------------------------------
LOAN_FEATURES = [
    "loan_amnt",
    "term",
]

BORROWER_FEATURES = [
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
]

FICO_FEATURES = [
    "fico_range_low",
    "fico_range_high",
]

CREDIT_DEPTH_FEATURES = [
    "open_acc",
    "total_acc",
    "num_sats",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mort_acc",
]

DELINQUENCY_FEATURES = [
    "delinq_2yrs",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "num_accts_ever_120_pd",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
]

INQUIRY_FEATURES = [
    "inq_last_6mths",
    "mths_since_recent_inq",
    "acc_open_past_24mths",
    "mo_sin_rcnt_tl",
    "mo_sin_rcnt_rev_tl_op",
    "num_tl_op_past_12m",
    "mths_since_recent_bc",
]

REVOLVING_FEATURES = [
    "revol_bal",
    "revol_util",
    "total_rev_hi_lim",
    "num_rev_accts",
    "num_actv_rev_tl",
    "num_rev_tl_bal_gt_0",
    "num_op_rev_tl",
]

BANKCARD_FEATURES = [
    "bc_open_to_buy",
    "bc_util",
    "num_actv_bc_tl",
    "num_bc_sats",
    "num_bc_tl",
    "percent_bc_gt_75",
]

BALANCE_FEATURES = [
    "tot_cur_bal",
    "avg_cur_bal",
    "total_bal_ex_mort",
    "tot_hi_cred_lim",
    "total_bc_limit",
    "num_il_tl",
    "total_il_high_credit_limit",
]

PUBLIC_RECORD_FEATURES = [
    "pub_rec",
    "pub_rec_bankruptcies",
    "tax_liens",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
    "chargeoff_within_12_mths",
]

FEATURE_COLUMNS = (
    LOAN_FEATURES
    + BORROWER_FEATURES
    + FICO_FEATURES
    + CREDIT_DEPTH_FEATURES
    + DELINQUENCY_FEATURES
    + INQUIRY_FEATURES
    + REVOLVING_FEATURES
    + BANKCARD_FEATURES
    + BALANCE_FEATURES
    + PUBLIC_RECORD_FEATURES
)

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET]


def main() -> None:
    """Load raw Lending Club data, filter, clean, and save."""

    print(f"Reading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, low_memory=False, usecols=ALL_COLUMNS)
    print(f"  Raw rows loaded: {len(df):,}")

    # Keep only terminal loan statuses
    df = df[df[TARGET].isin(KEEP_STATUSES)].copy()
    print(f"  Rows after status filter (Fully Paid / Charged Off): {len(df):,}")

    # Clean whitespace from term (" 36 months" -> "36 months")
    df["term"] = df["term"].str.strip()

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Summary stats
    n_total = len(df)
    n_default = (df[TARGET] == "Charged Off").sum()
    default_rate = n_default / n_total if n_total > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"  Rows:          {n_total:,}")
    print(f"  Features:      {len(FEATURE_COLUMNS)}")
    print(f"  Charged Off:   {n_default:,} ({default_rate:.2%})")
    print(f"  Fully Paid:    {n_total - n_default:,} ({1 - default_rate:.2%})")
    print(f"  Columns saved: {list(df.columns)}")


if __name__ == "__main__":
    main()
