def underwrite(application: dict) -> dict:
    # 1. Extract Variables with Safe Defaults
    risk_est = application.get('ExternalRiskEstimate', -9)
    rev_burden = application.get('NetFractionRevolvingBurden', -9)
    inst_burden = application.get('NetFractionInstallBurden', -9)
    num_trades_60 = application.get('NumTrades60Ever2DerogPubRec', 0)
    num_trades_90 = application.get('NumTrades90Ever2DerogPubRec', 0)
    num_sat_trades = application.get('NumSatisfactoryTrades', -9)
    total_trades = application.get('NumTotalTrades', 0)
    perc_never_delq = application.get('PercentTradesNeverDelq', -9)
    months_oldest = application.get('MSinceOldestTradeOpen', -9)
    num_inq = application.get('NumInqLast6M', 0)
    num_open_12m = application.get('NumTradesOpeninLast12M', 0)
    max_delq_12m = application.get('MaxDelq2PublicRecLast12M', 7) 
    msince_delq = application.get('MSinceMostRecentDelq', -9)
    num_high_util = application.get('NumBank2NatlTradesWHighUtilization', 0)
    pct_trades_bal = application.get('PercentTradesWBalance', -9)

    # 2. Decision Logic
    decision = "APPROVE"
    
    # Missing Data
    if risk_est == -9:
        decision = "DENY"
        
    # Hard Score Floor
    elif risk_est < 65:
        decision = "DENY"

    # Marginal Band (65 - 69)
    elif risk_est < 70:
        # Requires deep file and clean history
        if num_sat_trades < 20: 
            decision = "DENY"
        elif num_trades_60 > 0:
            decision = "DENY"
        elif rev_burden > 50:
            decision = "DENY"
        elif perc_never_delq != -9 and perc_never_delq < 100:
            decision = "DENY"

    # Low-Good Band (70 - 74)
    elif risk_est < 75:
        if num_trades_60 > 1:
            decision = "DENY"
        elif num_trades_60 == 1 and num_sat_trades < 30:
            decision = "DENY"
        elif perc_never_delq != -9 and perc_never_delq < 95:
            decision = "DENY"
        elif rev_burden > 70:
            decision = "DENY"
        elif num_inq > 3 and num_sat_trades < 35:
             decision = "DENY"
        elif msince_delq != -9 and msince_delq > -1 and msince_delq < 12:
            decision = "DENY"

    # High-Good Band (75 - 79)
    elif risk_est < 80:
        if rev_burden > 85 and num_inq > 3:
            decision = "DENY"
        elif num_sat_trades < 12 and num_inq > 2:
            decision = "DENY"
        elif rev_burden == -8 and total_trades < 5:
            decision = "DENY"

    # Excellent Band (80+)
    else:
        if rev_burden > 100:
             decision = "DENY"

    # 3. Reason Code Generation
    reasons = []

    if risk_est == -9:
        reasons = [
            "No bureau record or public record found",
            "Too few accounts currently paid as agreed",
            "Length of time accounts have been established"
        ]
    else:
        # Delinquency
        if (perc_never_delq != -9 and perc_never_delq < 100) or num_trades_60 > 0:
            reasons.append("Delinquent past or present on accounts")
        
        if num_trades_90 > 0 or (max_delq_12m < 7 and decision == "DENY"):
            reasons.append("Serious delinquency or public record")
            
        if num_trades_60 > 1:
            reasons.append("Number of accounts with delinquency")

        # Utilization
        if rev_burden != -9 and rev_burden != -8:
            if rev_burden > 40:
                reasons.append("Proportion of balances to credit limits too high")
            elif rev_burden > 20:
                reasons.append("Proportion of balances to credit limits could be lower")
            
            if rev_burden > 50 or (rev_burden > 30 and num_high_util > 0):
                reasons.append("Amount owed on accounts is too high")
                
        if num_high_util > 0:
            reasons.append("Number of accounts with high utilization")

        # Installment
        if inst_burden != -9 and inst_burden != -8:
            if inst_burden > 50:
                reasons.append("Installment loan balances relative to original amounts are high")
            elif inst_burden > 25:
                reasons.append("Installment loan balances relative to original amounts are moderate")

        # Age
        if months_oldest != -9 and months_oldest < 60:
            reasons.append("Length of time accounts have been established")

        # Thin / Count
        if num_sat_trades != -9 and num_sat_trades <= 5:
            reasons.append("Too few accounts currently paid as agreed")
            
        if rev_burden == -8 and decision == "DENY":
            reasons.append("Too few revolving accounts")

        # Inquiries
        if num_inq > 0 or num_open_12m > 1:
            reasons.append("Too many accounts recently opened")

        # Balances Pct
        if pct_trades_bal != -9:
            if pct_trades_bal > 75:
                reasons.append("Percentage of accounts carrying balances is too high")
            elif pct_trades_bal > 50:
                reasons.append("Percentage of accounts carrying balances is elevated")

    # Clean reasons
    unique_reasons = []
    seen = set()
    for r in reasons:
        if r not in seen:
            unique_reasons.append(r)
            seen.add(r)
            
    if decision == "APPROVE" and not unique_reasons:
        unique_reasons.append("No significant risk factors identified")
        
    score_val = 100.0 - float(risk_est) if risk_est != -9 else 100.0

    return {
        "decision": decision,
        "reason_codes": unique_reasons[:4],
        "score": score_val
    }