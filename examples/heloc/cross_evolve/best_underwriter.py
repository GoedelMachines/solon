def underwrite(application: dict) -> dict:
    """
    Deterministic credit underwriting function for evaluating loan applications.
    Based on tiered logic around ExternalRiskEstimate and secondary credit factors.
    """
    
    # 1. Helper for safe data access
    def get(key):
        return application.get(key, -9)

    # 2. Extract specific variables used in logic
    ere = get("ExternalRiskEstimate")
    rev_burden = get("NetFractionRevolvingBurden")
    inst_burden = get("NetFractionInstallBurden")
    pct_never_delq = get("PercentTradesNeverDelq")
    sat_trades = get("NumSatisfactoryTrades")
    total_trades = get("NumTotalTrades")
    ms_oldest = get("MSinceOldestTradeOpen")
    avg_age = get("AverageMInFile")
    inq_6m = get("NumInqLast6M")
    open_12m = get("NumTradesOpeninLast12M")
    ms_recent_delq = get("MSinceMostRecentDelq")
    max_delq_12m = get("MaxDelq2PublicRecLast12M")
    num_derog = get("NumTrades60Ever2DerogPubRec")
    high_util_trades = get("NumBank2NatlTradesWHighUtilization")
    pct_trades_w_bal = get("PercentTradesWBalance")

    # 3. Calculate Score (Lower is better risk, 0-100 scale)
    if ere > 0:
        score = float(100 - ere)
    else:
        score = 100.0

    # 4. Decision Logic
    decision = "APPROVE"
    reasons = []
    
    # Case: Missing Bureau Record
    if ere == -9:
        decision = "DENY"
        reasons = [
            "No bureau record or public record found",
            "Too few accounts currently paid as agreed",
            "Length of time accounts have been established"
        ]
    
    # Case: High Risk Tier
    elif ere < 64:
        decision = "DENY"
        
    # Case: Marginal Tier (64-69)
    # Requires deep, clean file to pass
    elif ere < 70:
        if pct_never_delq < 100 or sat_trades < 20 or ms_oldest < 100 or rev_burden > 50:
            decision = "DENY"

    # Case: Fair Tier (70-74)
    # Deny for recent delinquency or high leverage
    elif ere < 75:
        if pct_never_delq < 95 or sat_trades < 15 or (ms_recent_delq >= 0 and ms_recent_delq < 12) or rev_burden > 80:
            decision = "DENY"

    # Case: Good Tier (75-79)
    # Deny for thin files or excessive inquiries combined with thin file
    elif ere < 80:
        if (sat_trades < 15 and inq_6m > 2) or (sat_trades < 5 and rev_burden == -8) or rev_burden > 95:
            decision = "DENY"

    # Case: Excellent Tier (80+)
    # Deny only for extremely thin files (1 trade, no revolving)
    else:
        if total_trades <= 1 and rev_burden == -8:
            decision = "DENY"

    # Global Safety Net: Severe historical delinquency
    if pct_never_delq < 70 and pct_never_delq != -9:
        decision = "DENY"

    # 5. Reason Code Generation
    
    if decision == "DENY" and not reasons: # (Unless already set by ERE -9)
        # Delinquency
        if pct_never_delq < 100 or (ms_recent_delq >= 0 and ms_recent_delq < 24):
            reasons.append("Delinquent past or present on accounts")
        
        if num_derog > 0 or (max_delq_12m >= 0 and max_delq_12m < 7) or ere < 65:
            reasons.append("Serious delinquency or public record")
            
        # Utilization / Burden
        if rev_burden > 50 or rev_burden == -8:
            reasons.append("Proportion of balances to credit limits too high")
            if rev_burden == -8:
                reasons.append("Too few revolving accounts")
        elif pct_trades_w_bal > 75:
            reasons.append("Percentage of accounts carrying balances is too high")
            
        if high_util_trades >= 1:
            reasons.append("Number of accounts with high utilization")
            
        # History Length
        if ms_oldest < 120 or avg_age < 60:
            reasons.append("Length of time accounts have been established")
            
        # File Depth
        if sat_trades < 15:
            reasons.append("Too few accounts currently paid as agreed")
            
        # Inquiries
        if inq_6m > 1:
            reasons.append("Too many accounts recently opened")
        
        # Fallback
        if not reasons:
            reasons.append("External risk estimate is too low")

    elif decision == "APPROVE":
        # Approval Warnings
        if inq_6m > 0 and (open_12m > 0 or ere < 75):
            reasons.append("Too many accounts recently opened")
            
        if rev_burden == -8:
             reasons.append("Proportion of balances to credit limits too high")
        elif rev_burden > 50:
             reasons.append("Proportion of balances to credit limits too high")
        elif rev_burden > 30:
            reasons.append("Proportion of balances to credit limits could be lower")
            
        if inst_burden > 50:
            reasons.append("Installment loan balances relative to original amounts are high")
        elif inst_burden > 20:
            reasons.append("Installment loan balances relative to original amounts are moderate")
            
        if pct_never_delq < 100:
            reasons.append("Delinquent past or present on accounts")
            
        if ms_oldest < 150 and ere < 85:
            reasons.append("Length of time accounts have been established")
            
        if pct_trades_w_bal > 65:
            reasons.append("Percentage of accounts carrying balances is elevated")
        
        if sat_trades < 25:
            reasons.append("Too few accounts currently paid as agreed")
            
        if high_util_trades >= 1:
            reasons.append("Number of accounts with high utilization")
            
        if not reasons:
            reasons.append("No significant risk factors identified")

    # 6. Formatting
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    return {
        "decision": decision,
        "reason_codes": unique_reasons[:4],
        "score": round(score, 2)
    }