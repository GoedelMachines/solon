def underwrite(application: dict) -> dict:
    """
    Evaluates credit applications based on deterministic rules examining risk scores,
    payment history, credit utilization, and credit file depth.
    """

    # --- Helper: Data Extraction and Missing Value Handling ---
    # Special values: 
    # -9: No Bureau Record / Missing
    # -8: No Revolving Accounts / No Balance
    # -7: Not Applicable / No Delinquency
    def get_v(key, default_val=0):
        val = application.get(key, -9)
        if val in [-7, -8, -9]:
            return default_val
        return val

    # Primary Risk Metrics
    ere = application.get("ExternalRiskEstimate", -9)
    nfrb_raw = application.get("NetFractionRevolvingBurden", -9)
    nfrb_val = get_v("NetFractionRevolvingBurden", 0)
    ptnd = get_v("PercentTradesNeverDelq", 100)
    nst = get_v("NumSatisfactoryTrades", 0)
    moto = get_v("MSinceOldestTradeOpen", 0)
    inq6m = get_v("NumInqLast6M", 0)
    open12m = get_v("NumTradesOpeninLast12M", 0)
    high_util_trades = get_v("NumBank2NatlTradesWHighUtilization", 0)
    install_burden = get_v("NetFractionInstallBurden", 0)
    ptwb = get_v("PercentTradesWBalance", 0)
    
    # Delinquency Recency: handle missing values by assigning a high number (representing no recent delq)
    msrd_raw = application.get("MSinceMostRecentDelq", -7)
    msrd = msrd_raw if msrd_raw >= 0 else 999
    
    derog60 = get_v("NumTrades60Ever2DerogPubRec", 0)
    max_delq_12 = get_v("MaxDelq2PublicRecLast12M", 7) # 7 is the scale for 'No Delinquency'

    # --- Business Decision Logic ---
    decision = "APPROVE"
    
    # Rule 1: Fatal Flaws - No record or critically low risk estimate (sub-prime range)
    if ere == -9:
        decision = "DENY"
    elif ere < 65:
        decision = "DENY"
        
    # Rule 2: Risk Score 65-69 (Below Median / High Scrutiny)
    # Requires an established file and high payment integrity to be approved.
    elif ere < 70:
        if ptnd < 98:
            decision = "DENY"  # Recent or significant delinquency
        elif nst < 25:
            decision = "DENY"  # Insufficient account depth
        elif nfrb_val > 60 or nfrb_raw == -8:
            decision = "DENY"  # High utilization or lack of revolving history
            
    # Rule 3: Risk Score 70-74 (Near Median)
    # Sensitivity to extreme utilization and recent credit events.
    elif ere < 75:
        if nfrb_val > 90 and nst < 25:
            decision = "DENY"  # Maxed utilization on moderate file
        elif ptnd < 92:
            decision = "DENY"  # Historical delinquency concerns
        elif msrd < 6:
            decision = "DENY"  # Very recent delinquency
            
    # Rule 4: Risk Score 75-81 (Prime Range)
    # Focus shifts toward protecting against "thin file" applicants or aggressive credit seeking.
    elif ere < 82:
        if nfrb_raw == -8 and nst < 5:
            decision = "DENY"  # Thin file with only installment loans (Case 79)
        elif inq6m >= 3 and nst < 15:
            decision = "DENY"  # Significant recent inquiries relative to file depth

    # --- Reason Code Generation ---
    reasons = []

    if ere == -9:
        # Standard reasons for no record found
        reasons = [
            "No bureau record or public record found",
            "Too few accounts currently paid as agreed",
            "Length of time accounts have been established"
        ]
    else:
        # Priority 1: Serious Delinquencies & Public Records
        if derog60 > 0 or msrd < 12 or max_delq_12 < 6:
            reasons.append("Serious delinquency or public record")
        
        # Priority 2: General Payment History
        if ptnd < 100:
            reasons.append("Delinquent past or present on accounts")
        
        # Priority 3: Utilization and Balance Burden
        # Handling the -8 (No revolving) case specifically for reason code generation
        if nfrb_raw == -8 or nfrb_val > 60:
            reasons.append("Proportion of balances to credit limits too high")
        elif nfrb_val > 25:
            reasons.append("Proportion of balances to credit limits could be lower")
            
        if high_util_trades > 0:
            reasons.append("Number of accounts with high utilization")
            
        if nfrb_val > 50 or install_burden > 60:
            reasons.append("Amount owed on accounts is too high")
        
        if install_burden > 75:
            reasons.append("Installment loan balances relative to original amounts are high")
        elif install_burden > 50:
            reasons.append("Installment loan balances relative to original amounts are moderate")

        # Priority 4: Credit Maturity and Account History
        if nst < 20:
            reasons.append("Too few accounts currently paid as agreed")
        if moto < 120:
            reasons.append("Length of time accounts have been established")
        if nfrb_raw == -8:
            reasons.append("Too few revolving accounts")
            
        # Priority 5: Recent Credit Activity and Current Exposure
        if inq6m > 0 or open12m > 0:
            reasons.append("Too many accounts recently opened")
        if ptwb > 75:
            reasons.append("Percentage of accounts carrying balances is too high")
        elif ptwb > 50:
            reasons.append("Percentage of accounts carrying balances is elevated")

    # Clean and prioritize up to 4 reasons
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)
    
    # Compliance Fallbacks
    if decision == "APPROVE" and not unique_reasons:
        unique_reasons = ["No significant risk factors identified"]
    if decision == "DENY" and not unique_reasons:
        unique_reasons = ["Risk estimate too low for credit profile"]

    # --- Score Calculation (Lower = Less Risky) ---
    if ere == -9:
        base_score = 95.0
    else:
        # Scale based on inverted risk estimate (100 - ERE)
        base_score = 100.0 - float(ere)
    
    # Minor deterministic adjustments based on utilization, inquiries, and history
    util_penalty = nfrb_val / 10.0
    inq_penalty = inq6m * 1.5
    history_penalty = (100 - ptnd) / 4.0
    
    final_score = base_score + util_penalty + inq_penalty + history_penalty

    return {
        "decision": decision,
        "reason_codes": unique_reasons[:4],
        "score": round(max(0.0, min(100.0, final_score)), 1)
    }