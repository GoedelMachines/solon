"""
Generate synthetic HELOC dataset matching the real FICO HELOC schema.
Replace with real data from: https://huggingface.co/datasets/mstz/heloc

Real dataset: 10,459 records, 23 features + 1 target.
All features are integer-valued risk metrics from credit bureau data.
Values of -7, -8, -9 indicate special conditions (no trades, no usable data, etc.)
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 10459  # Match real dataset size


def generate_feature(name, low, high, special_pct=0.05):
    """Generate a feature with some special values (-7, -8, -9)."""
    n_special = int(N * special_pct)
    n_normal = N - n_special
    normal_vals = np.random.randint(low, high + 1, size=n_normal)
    special_vals = np.random.choice([-7, -8, -9], size=n_special)
    vals = np.concatenate([normal_vals, special_vals])
    np.random.shuffle(vals)
    return vals


data = {
    # External risk estimate (like FICO score proxy, 0-100 scale)
    "ExternalRiskEstimate": generate_feature("ere", 40, 94, special_pct=0.01),

    # Months since oldest trade open
    "MSinceOldestTradeOpen": generate_feature("msoto", 12, 600, special_pct=0.03),

    # Months since most recent trade open
    "MSinceMostRecentTradeOpen": generate_feature("msmreto", 0, 120, special_pct=0.03),

    # Average months in file
    "AverageMInFile": generate_feature("amif", 6, 350, special_pct=0.03),

    # Number of satisfactory trades
    "NumSatisfactoryTrades": generate_feature("nst", 0, 79, special_pct=0.02),

    # Number of trades opened in last 60 months
    "NumTrades60Ever2DerogPubRec": generate_feature("nt60", 0, 19, special_pct=0.03),

    # Number of trades opened in last 90 days
    "NumTrades90Ever2DerogPubRec": generate_feature("nt90", 0, 19, special_pct=0.03),

    # Percent trades never delinquent
    "PercentTradesNeverDelq": generate_feature("ptnd", 0, 100, special_pct=0.02),

    # Months since most recent delinquency
    "MSinceMostRecentDelq": generate_feature("msmrd", 0, 83, special_pct=0.15),

    # Max delinquency last 12 months (0-7 scale)
    "MaxDelq2PublicRecLast12M": generate_feature("md12", 0, 9, special_pct=0.02),

    # Max delinquency ever (1-9 scale)
    "MaxDelqEver": generate_feature("mde", 1, 9, special_pct=0.01),

    # Number of total trades
    "NumTotalTrades": generate_feature("ntt", 0, 104, special_pct=0.01),

    # Number of trades opened in last 12 months
    "NumTradesOpeninLast12M": generate_feature("ntol12", 0, 19, special_pct=0.02),

    # Percent installment trades
    "PercentInstallTrades": generate_feature("pit", 0, 100, special_pct=0.03),

    # Months since most recent inquiry (excluding last 7 days)
    "MSinceMostRecentInqexcl7days": generate_feature("msmri", 0, 24, special_pct=0.15),

    # Number of inquiries in last 6 months
    "NumInqLast6M": generate_feature("nil6", 0, 66, special_pct=0.02),

    # Number of inquiries in last 6 months (excluding last 7 days)
    "NumInqLast6Mexcl7days": generate_feature("nil6e7", 0, 66, special_pct=0.02),

    # Net fraction revolving burden (revolving balance / credit limit)
    "NetFractionRevolvingBurden": generate_feature("nfrb", 0, 232, special_pct=0.05),

    # Net fraction installment burden
    "NetFractionInstallBurden": generate_feature("nfib", 0, 471, special_pct=0.08),

    # Number of revolving trades with balance
    "NumRevolvingTradesWBalance": generate_feature("nrtwb", 0, 32, special_pct=0.03),

    # Number of installment trades with balance
    "NumInstallTradesWBalance": generate_feature("nitwb", 0, 23, special_pct=0.03),

    # Number of bank/national trades with high utilization ratio
    "NumBank2NatlTradesWHighUtilization": generate_feature("nb2ntwhu", 0, 18, special_pct=0.03),

    # Percent trades with balance
    "PercentTradesWBalance": generate_feature("ptwb", 0, 100, special_pct=0.03),
}

df = pd.DataFrame(data)

# Generate target: RiskPerformance (Good=0, Bad=1)
# Create a realistic relationship: higher risk estimate = lower default probability
risk_score = (
    -0.05 * df["ExternalRiskEstimate"].clip(0, 100)
    + 0.02 * df["NumTrades60Ever2DerogPubRec"].clip(0, 20)
    + 0.02 * df["NumTrades90Ever2DerogPubRec"].clip(0, 20)
    - 0.01 * df["PercentTradesNeverDelq"].clip(0, 100)
    + 0.03 * df["NetFractionRevolvingBurden"].clip(0, 250)
    + 0.01 * df["NumInqLast6M"].clip(0, 70)
    - 0.005 * df["NumSatisfactoryTrades"].clip(0, 80)
    + 0.02 * df["MaxDelqEver"].clip(1, 9)
    + np.random.normal(0, 0.5, N)
)
prob_bad = 1 / (1 + np.exp(-risk_score))
df["RiskPerformance"] = (np.random.random(N) < prob_bad).astype(int)

# Ensure roughly 52% bad rate (matching real dataset)
target_bad_rate = 0.52
current_bad_rate = df["RiskPerformance"].mean()
if current_bad_rate < target_bad_rate:
    flip_n = int((target_bad_rate - current_bad_rate) * N)
    good_idx = df[df["RiskPerformance"] == 0].sample(flip_n).index
    df.loc[good_idx, "RiskPerformance"] = 1
elif current_bad_rate > target_bad_rate:
    flip_n = int((current_bad_rate - target_bad_rate) * N)
    bad_idx = df[df["RiskPerformance"] == 1].sample(flip_n).index
    df.loc[bad_idx, "RiskPerformance"] = 0

print(f"Dataset shape: {df.shape}")
print(f"Bad rate: {df['RiskPerformance'].mean():.2%}")
print(f"Good: {(df['RiskPerformance']==0).sum()}, Bad: {(df['RiskPerformance']==1).sum()}")
print(f"\nFeature summary:")
print(df.describe().T[["mean", "std", "min", "max"]].to_string())

import os
output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "heloc_synthetic.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
