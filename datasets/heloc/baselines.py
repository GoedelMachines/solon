"""XGBoost baselines for the HELOC results in the README.

Needs the data from fetch_heloc.py and `pip install xgboost`.

- full data: 5-fold stratified CV AUC on all rows;
- n=100: trained on the 100 example cases only (data/heloc_100_cases.csv), scored on every
  row not identical to one of them (identical rows, mostly the all-missing -9 records, are
  dropped so no test row was seen in training).

Usage:
    python datasets/heloc/baselines.py [--data data/heloc_real.csv] [--cases data/heloc_100_cases.csv]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "RiskPerformance"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/heloc_real.csv")
    ap.add_argument("--cases", default="data/heloc_100_cases.csv")
    args = ap.parse_args()
    from xgboost import XGBClassifier

    full, cases = pd.read_csv(args.data), pd.read_csv(args.cases)
    X, y = full.drop(columns=TARGET), full[TARGET]

    aucs = []
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                              eval_metric="logloss", random_state=0)
        model.fit(X.iloc[train], y.iloc[train])
        aucs.append(roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]))
    print(f"full data ({len(full):,} rows), 5-fold CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    seen = pd.merge(full.reset_index(), cases, on=list(cases.columns))["index"]
    rest = full.drop(index=seen.unique())
    aucs = []
    for seed in range(5):
        model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8,
                              eval_metric="logloss", random_state=seed)
        model.fit(cases.drop(columns=TARGET), cases[TARGET])
        aucs.append(roc_auc_score(rest[TARGET], model.predict_proba(rest.drop(columns=TARGET))[:, 1]))
    print(f"n=100 example cases -> {len(rest):,} held-out rows, AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


if __name__ == "__main__":
    main()
