"""Fetch the FICO HELOC dataset and rebuild the 100 example cases.

The dataset itself is not redistributed in this repository. This script:

1. loads it from a CSV you downloaded (``--input``, e.g. FICO's
   ``heloc_dataset_v1.csv`` or the Kaggle copy), or from the Hugging Face
   hub (``mstz/heloc``, needs ``pip install datasets``);
2. normalizes it to the layout the pipeline expects (23 integer features,
   ``RiskPerformance`` last, Bad=1 / Good=0) -> ``data/heloc_real.csv``;
3. re-selects the 100 cases used by ``examples/heloc`` by fingerprint
   (``example_cases.json``) -> ``data/heloc_100_cases.csv``, in the
   ``case_index`` order the shipped targets refer to.

Usage:
    python datasets/heloc/fetch_heloc.py [--input heloc_dataset_v1.csv]
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "RiskPerformance"


# mstz/heloc on Hugging Face is the same 10,459 rows and 24 columns, renamed; is_at_risk=1 is Bad.
MSTZ_COLUMNS = {
    "estimate_of_risk": "ExternalRiskEstimate",
    "months_since_first_trade": "MSinceOldestTradeOpen",
    "months_since_last_trade": "MSinceMostRecentTradeOpen",
    "average_duration_of_resolution": "AverageMInFile",
    "number_of_satisfactory_trades": "NumSatisfactoryTrades",
    "nr_trades_insolvent_for_over_60_days": "NumTrades60Ever2DerogPubRec",
    "nr_trades_insolvent_for_over_90_days": "NumTrades90Ever2DerogPubRec",
    "percentage_of_legal_trades": "PercentTradesNeverDelq",
    "months_since_last_illegal_trade": "MSinceMostRecentDelq",
    "maximum_illegal_trades_over_last_year": "MaxDelq2PublicRecLast12M",
    "maximum_illegal_trades": "MaxDelqEver",
    "nr_total_trades": "NumTotalTrades",
    "nr_trades_initiated_in_last_year": "NumTradesOpeninLast12M",
    "percentage_of_installment_trades": "PercentInstallTrades",
    "months_since_last_inquiry_not_recent": "MSinceMostRecentInqexcl7days",
    "nr_inquiries_in_last_6_months": "NumInqLast6M",
    "nr_inquiries_in_last_6_months_not_recent": "NumInqLast6Mexcl7days",
    "net_fraction_of_revolving_burden": "NetFractionRevolvingBurden",
    "net_fraction_of_installment_burden": "NetFractionInstallBurden",
    "nr_revolving_trades_with_balance": "NumRevolvingTradesWBalance",
    "nr_installment_trades_with_balance": "NumInstallTradesWBalance",
    "nr_banks_with_high_ratio": "NumBank2NatlTradesWHighUtilization",
    "percentage_trades_with_balance": "PercentTradesWBalance",
    "is_at_risk": TARGET,
}


def load_raw(path):
    if path:
        df = pd.read_csv(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("Pass --input <csv>, or `pip install datasets` to fetch mstz/heloc from Hugging Face.")
        df = load_dataset("mstz/heloc")["train"].to_pandas()
    if set(MSTZ_COLUMNS) <= set(df.columns):
        df = df.rename(columns=MSTZ_COLUMNS)
    return df


def normalize(df, columns):
    features = [c for c in columns if c != TARGET]
    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing HELOC feature columns: {missing}")
    if TARGET not in df.columns:
        extra = [c for c in df.columns if c not in features]
        if len(extra) != 1:
            sys.exit(f"Cannot identify the outcome column among {extra}; rename it to {TARGET}.")
        df = df.rename(columns={extra[0]: TARGET})
    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.str.strip().str.lower().map({"bad": 1, "good": 0})
    df = df.assign(**{TARGET: y.astype(int)})
    return df[columns].astype(int)


def fingerprint(values):
    return hashlib.sha256(",".join(str(int(v)) for v in values).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="Path to a downloaded HELOC CSV (otherwise Hugging Face mstz/heloc)")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    spec = json.load(open(os.path.join(HERE, "example_cases.json")))
    df = normalize(load_raw(args.input), spec["columns"])
    os.makedirs(args.out_dir, exist_ok=True)
    real_path = os.path.join(args.out_dir, "heloc_real.csv")
    df.to_csv(real_path, index=False)
    print(f"{real_path}: {len(df):,} rows, bad rate {df[TARGET].mean():.1%}")

    first_row = {}
    for i, row in enumerate(df.itertuples(index=False)):
        first_row.setdefault(fingerprint(row), i)
    rows = [first_row.get(h) for h in spec["cases"]]
    found = sum(r is not None for r in rows)
    if found != len(rows):
        sys.exit(f"Only {found}/{len(rows)} example cases found; this copy of the dataset differs. "
                 f"Use `solon sample` to draw a fresh sample instead.")
    cases_path = os.path.join(args.out_dir, "heloc_100_cases.csv")
    df.iloc[rows].to_csv(cases_path, index=False)
    print(f"{cases_path}: {len(rows)} example cases (matching examples/heloc targets)")


if __name__ == "__main__":
    main()
