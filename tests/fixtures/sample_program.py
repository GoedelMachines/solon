import numpy as np


def underwrite(application: dict) -> dict:
    """Simple test underwriting function."""
    def get(feature, default=None):
        val = application.get(feature)
        if val in (-7, -8, -9) or val is None:
            return default
        return val

    risk_est = get("ExternalRiskEstimate", 50)
    rev_burden = get("NetFractionRevolvingBurden", 50)
    derog = get("NumTrades60Ever2DerogPubRec", 0)

    score = 100 - risk_est + rev_burden * 0.5 + derog * 10
    reason_codes = []

    if risk_est < 60:
        reason_codes.append("Low credit risk score")
    if rev_burden > 80:
        reason_codes.append("High revolving utilization")
    if derog > 2:
        reason_codes.append("Too many delinquent accounts")

    if risk_est >= 70 and rev_burden < 60 and derog <= 1:
        decision = "APPROVE"
        if not reason_codes:
            reason_codes.append("Acceptable credit profile")
    else:
        decision = "DENY"
        if not reason_codes:
            reason_codes.append("Overall credit risk too high")

    return {"decision": decision, "reason_codes": reason_codes, "score": score}
