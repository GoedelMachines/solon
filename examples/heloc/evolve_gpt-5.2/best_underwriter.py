def underwrite(application: dict) -> dict:
    # Helper: fetch integer field and normalize special missing values (-7, -8, -9) to None
    def get(name: str):
        value = application.get(name, None)
        if value in (-7, -8, -9, None):
            return None
        try:
            return int(value)
        except Exception:
            return None

    # -----------------------------
    # Pull variables (missing-safe)
    # -----------------------------
    external_risk = get("ExternalRiskEstimate")

    months_since_oldest_trade = get("MSinceOldestTradeOpen")
    average_months_in_file = get("AverageMInFile")

    num_satisfactory_trades = get("NumSatisfactoryTrades")
    num_total_trades = get("NumTotalTrades")
    num_opened_last_12m = get("NumTradesOpeninLast12M")

    num_60dpd_or_derog = get("NumTrades60Ever2DerogPubRec")
    num_90dpd_or_derog = get("NumTrades90Ever2DerogPubRec")
    percent_never_delinquent = get("PercentTradesNeverDelq")
    months_since_most_recent_delq = get("MSinceMostRecentDelq")
    max_delq_last_12m = get("MaxDelq2PublicRecLast12M")
    max_delq_ever = get("MaxDelqEver")

    num_inq_last_6m = get("NumInqLast6M")
    num_inq_last_6m_excl_7d = get("NumInqLast6Mexcl7days")
    months_since_most_recent_inq = get("MSinceMostRecentInqexcl7days")

    revolving_util = get("NetFractionRevolvingBurden")
    installment_util = get("NetFractionInstallBurden")

    num_revolving_with_balance = get("NumRevolvingTradesWBalance")
    num_install_with_balance = get("NumInstallTradesWBalance")
    num_high_util_trades = get("NumBank2NatlTradesWHighUtilization")
    percent_trades_with_balance = get("PercentTradesWBalance")

    # --------------------------------------------
    # Reason helper (de-dupe, preserve order)
    # --------------------------------------------
    reason_codes = []

    def add_reason(code: str):
        if code and code not in reason_codes:
            reason_codes.append(code)

    # --------------------------------------------
    # 0) No bureau record / cannot score at all
    # --------------------------------------------
    # Deny when essentially all key bureau indicators are missing.
    core_fields = (
        external_risk,
        revolving_util,
        num_60dpd_or_derog,
        percent_never_delinquent,
        num_inq_last_6m,
    )
    if external_risk is None and sum(1 for v in core_fields if v is None) >= 4:
        add_reason("No bureau record or public record found")
        add_reason("Too few accounts currently paid as agreed")
        add_reason("Length of time accounts have been established")
        return {"decision": "DENY", "reason_codes": reason_codes[:4], "score": 95.0}

    # --------------------------------------------
    # 1) Interpretable flags (used for decisions)
    # --------------------------------------------
    # File depth
    extremely_thin_file = False
    if num_total_trades is not None and num_total_trades <= 1:
        extremely_thin_file = True
    if num_satisfactory_trades is not None and num_satisfactory_trades <= 1:
        extremely_thin_file = True

    thin_file = False
    if num_total_trades is not None and num_total_trades <= 2:
        thin_file = True
    if num_satisfactory_trades is not None and num_satisfactory_trades <= 2:
        thin_file = True

    # Established history
    young_file = False
    if months_since_oldest_trade is not None and months_since_oldest_trade < 60:
        young_file = True
    if average_months_in_file is not None and average_months_in_file < 48:
        young_file = True

    # Delinquency signals
    has_any_delinquency = False
    if num_60dpd_or_derog is not None and num_60dpd_or_derog > 0:
        has_any_delinquency = True
    if num_90dpd_or_derog is not None and num_90dpd_or_derog > 0:
        has_any_delinquency = True
    if percent_never_delinquent is not None and percent_never_delinquent < 90:
        has_any_delinquency = True

    # Recent/material delinquency
    recent_material_delq = False
    if max_delq_last_12m is not None and max_delq_last_12m >= 4:
        recent_material_delq = True
    if months_since_most_recent_delq is not None and months_since_most_recent_delq <= 12:
        # If there was a delinquency within a year, treat as material when severity is meaningful or unknown.
        if max_delq_last_12m is None or max_delq_last_12m >= 3:
            recent_material_delq = True

    # Special dataset nuance:
    # In this dataset, MaxDelqEver == 8 frequently appears on otherwise clean files.
    # We treat it as "unknown/neutral" rather than a delinquency.
    max_delq_ever_is_suspicious = False
    if max_delq_ever is not None and max_delq_ever not in (8, 0):
        # Some non-8 values do correlate with actual delinquency; use only as a weak signal.
        max_delq_ever_is_suspicious = max_delq_ever <= 5

    # Revolving utilization bands (percentage-like; can exceed 100)
    rev_util_over_limit = False
    rev_util_extreme = False
    rev_util_very_high = False
    rev_util_high = False
    rev_util_moderate = False

    if revolving_util is not None:
        if revolving_util >= 110:
            rev_util_over_limit = True
        if revolving_util >= 100:
            rev_util_extreme = True
            rev_util_very_high = True
            rev_util_high = True
        elif revolving_util >= 80:
            rev_util_very_high = True
            rev_util_high = True
        elif revolving_util >= 60:
            rev_util_high = True
        elif revolving_util >= 30:
            rev_util_moderate = True

    # Use inquiries excluding last 7 days if available
    inq6 = num_inq_last_6m_excl_7d if num_inq_last_6m_excl_7d is not None else num_inq_last_6m

    # --------------------------------------------
    # 2) Score (additive points; lower = less risky)
    # --------------------------------------------
    score = 28.0

    # ExternalRiskEstimate
    if external_risk is None:
        score += 10.0
    else:
        if external_risk >= 89:
            score -= 14.0
        elif external_risk >= 85:
            score -= 11.0
        elif external_risk >= 80:
            score -= 8.0
        elif external_risk >= 75:
            score -= 4.0
        elif external_risk >= 70:
            score += 1.0
        elif external_risk >= 65:
            score += 6.0
        elif external_risk >= 60:
            score += 12.0
        else:
            score += 20.0

    # Delinquency
    if has_any_delinquency or max_delq_ever_is_suspicious:
        score += 12.0
        add_reason("Delinquent past or present on accounts")

    if (num_60dpd_or_derog is not None and num_60dpd_or_derog >= 2) or (
        num_90dpd_or_derog is not None and num_90dpd_or_derog >= 1
    ):
        score += 8.0
        add_reason("Number of accounts with delinquency")

    if recent_material_delq:
        score += 18.0
        add_reason("Serious delinquency or public record")

    # Credit history age
    if young_file:
        score += 10.0
        add_reason("Length of time accounts have been established")

    # Satisfactory trades:
    # - Denials in the examples often cite this when truly thin.
    # - Approvals sometimes cite it informationally even with ~9-20 satisfactory trades.
    if num_satisfactory_trades is not None:
        if num_satisfactory_trades <= 1:
            score += 10.0
            add_reason("Too few accounts currently paid as agreed")
        elif num_satisfactory_trades <= 2:
            score += 7.0
            add_reason("Too few accounts currently paid as agreed")
        elif num_satisfactory_trades <= 4:
            score += 3.0
            add_reason("Too few accounts currently paid as agreed")
        elif num_satisfactory_trades <= 20:
            # Informational callout commonly seen in provided approvals.
            score += 1.0
            add_reason("Too few accounts currently paid as agreed")
    else:
        if thin_file:
            score += 4.0
            add_reason("Too few accounts currently paid as agreed")

    # Revolving utilization
    if revolving_util is not None:
        if revolving_util >= 100:
            score += 18.0
            add_reason("Proportion of balances to credit limits too high")
        elif revolving_util >= 80:
            score += 12.0
            add_reason("Proportion of balances to credit limits too high")
        elif revolving_util >= 60:
            score += 7.0
            add_reason("Proportion of balances to credit limits too high")
        elif revolving_util >= 30:
            score += 3.0
            add_reason("Proportion of balances to credit limits could be lower")
        else:
            score -= 2.0
    else:
        score += 1.0  # mild uncertainty

    # High utilization trades
    if num_high_util_trades is not None:
        if num_high_util_trades >= 2:
            score += 9.0
            add_reason("Number of accounts with high utilization")
        elif num_high_util_trades == 1:
            score += 4.0
            add_reason("Number of accounts with high utilization")

    # Amount owed proxy
    if revolving_util is not None and revolving_util >= 40:
        score += 4.0
        add_reason("Amount owed on accounts is too high")
    if (num_revolving_with_balance is not None and num_revolving_with_balance >= 6) or (
        num_install_with_balance is not None and num_install_with_balance >= 4
    ):
        score += 3.0
        add_reason("Amount owed on accounts is too high")

    # Percent trades with balance
    if percent_trades_with_balance is not None:
        if percent_trades_with_balance >= 80:
            score += 10.0
            add_reason("Percentage of accounts carrying balances is too high")
        elif percent_trades_with_balance >= 50:
            score += 4.0
            add_reason("Percentage of accounts carrying balances is elevated")

    # New credit activity (often a callout even on approvals)
    if num_opened_last_12m is not None:
        if num_opened_last_12m >= 3:
            score += 4.0
            add_reason("Too many accounts recently opened")
        elif num_opened_last_12m == 2:
            score += 2.0
            add_reason("Too many accounts recently opened")

    if inq6 is not None and inq6 >= 4:
        score += 2.0
        add_reason("Too many accounts recently opened")
    if months_since_most_recent_inq is not None and months_since_most_recent_inq <= 1:
        if inq6 is not None and inq6 >= 2:
            score += 1.0
            add_reason("Too many accounts recently opened")

    # Installment utilization (informational; appears on approvals)
    if installment_util is not None:
        if installment_util >= 80:
            score += 3.0
            add_reason("Installment loan balances relative to original amounts are high")
        elif installment_util >= 50:
            score += 2.0
            add_reason("Installment loan balances relative to original amounts are moderate")

    # Too few revolving accounts:
    # Target the known denial pattern: strong score but essentially no revolving history and missing utilization.
    too_few_revolving_accounts = False
    if revolving_util is None:
        if extremely_thin_file or thin_file:
            if num_revolving_with_balance is None or num_revolving_with_balance == 0:
                too_few_revolving_accounts = True
    if too_few_revolving_accounts:
        score += 6.0
        add_reason("Too few revolving accounts")

    # --------------------------------------------
    # 3) Explicit decision rules (deterministic)
    #    Fix the known misclassifications.
    # --------------------------------------------
    deny = False

    # A) Very thin + missing revolving utilization: deny (matches known denial pattern)
    if extremely_thin_file and revolving_util is None:
        deny = True

    # B) Over-limit utilization (110%+) is a denial
    if rev_util_over_limit:
        deny = True

    # C) Extreme utilization (100%+) is a denial when paired with weaker score or other pressure
    if rev_util_extreme:
        if external_risk is None or external_risk < 75:
            deny = True
        if has_any_delinquency or young_file or recent_material_delq:
            deny = True

    # D) Low external risk with compounding factors: deny
    if external_risk is not None and external_risk < 65:
        if rev_util_high or young_file or has_any_delinquency or recent_material_delq:
            deny = True

    # E) Recent material delinquency: generally deny unless exceptionally strong and otherwise clean
    if recent_material_delq:
        if external_risk is None or external_risk < 78:
            deny = True
        else:
            if rev_util_high or young_file:
                deny = True
            if percent_never_delinquent is not None and percent_never_delinquent < 95:
                deny = True

    # F) Key fixes from prior errors:
    #    1) Do NOT deny clean, deep files solely because delinquencies are missing (-7) or MaxDelqEver==8.
    clean_and_deep = False
    if external_risk is not None and external_risk >= 69:
        no_derogs_known = (num_60dpd_or_derog is not None and num_60dpd_or_derog == 0) and (
            num_90dpd_or_derog is None or num_90dpd_or_derog == 0
        )
        strong_never_delq = percent_never_delinquent is not None and percent_never_delinquent >= 95
        deep_file = (num_satisfactory_trades is not None and num_satisfactory_trades >= 10) or (
            num_total_trades is not None and num_total_trades >= 10
        )
        # months_since_most_recent_delq is often missing when none; treat missing as neutral for this dataset.
        if no_derogs_known and strong_never_delq and deep_file:
            clean_and_deep = True

    if clean_and_deep:
        # Keep only the "hard" denials.
        if not (rev_util_over_limit or (rev_util_extreme and (external_risk is None or external_risk < 75))):
            # Allow approval even if the score accumulator is high due to informational reasons.
            deny = False

    #    2) Deny when there is clear delinquency evidence with only middling score.
    #       (Catches: risk~69 + 60dpd>0 or neverdelq<90 or recent delinquency.)
    if (has_any_delinquency or recent_material_delq) and external_risk is not None and external_risk < 72:
        deny = True

    #    3) Deny when utilization is very high (>=90) AND there is any additional stress (inquiries, young file, thin file),
    #       even if otherwise "clean". (Catches: risk=78, util=90, inq=4; risk=71 util=95)
    if revolving_util is not None and revolving_util >= 90:
        additional_stress = False
        if inq6 is not None and inq6 >= 3:
            additional_stress = True
        if young_file or thin_file or extremely_thin_file:
            additional_stress = True
        if num_high_util_trades is not None and num_high_util_trades >= 2:
            additional_stress = True
        if additional_stress and not clean_and_deep:
            deny = True
        # Even with a deep file, keep a conservative decline for very high utilization >=95 unless score is very strong.
        if revolving_util >= 95 and (external_risk is None or external_risk < 80):
            deny = True

    #    4) Deny when file is young and utilization is high (matches: risk=65, util=76, young file)
    if young_file and revolving_util is not None and revolving_util >= 70:
        if external_risk is None or external_risk < 75:
            deny = True

    # Backstop: only deny at very high compounded risk
    if score >= 97.0:
        deny = True

    decision = "DENY" if deny else "APPROVE"

    # --------------------------------------------
    # 4) Reason code hygiene and ordering (1-4)
    # --------------------------------------------
    if decision == "APPROVE" and not reason_codes:
        reason_codes = ["No significant risk factors identified"]

    if decision == "DENY" and not reason_codes:
        add_reason("Delinquent past or present on accounts")

    # Priority ordering
    if decision == "DENY":
        priority = [
            "No bureau record or public record found",
            "Serious delinquency or public record",
            "Delinquent past or present on accounts",
            "Number of accounts with delinquency",
            "Proportion of balances to credit limits too high",
            "Number of accounts with high utilization",
            "Amount owed on accounts is too high",
            "Percentage of accounts carrying balances is too high",
            "Percentage of accounts carrying balances is elevated",
            "Too few accounts currently paid as agreed",
            "Length of time accounts have been established",
            "Too few revolving accounts",
            "Too many accounts recently opened",
            "Installment loan balances relative to original amounts are high",
            "Installment loan balances relative to original amounts are moderate",
            "Proportion of balances to credit limits could be lower",
            "No significant risk factors identified",
        ]
    else:
        priority = [
            "Too many accounts recently opened",
            "Proportion of balances to credit limits too high",
            "Proportion of balances to credit limits could be lower",
            "Number of accounts with high utilization",
            "Amount owed on accounts is too high",
            "Percentage of accounts carrying balances is too high",
            "Percentage of accounts carrying balances is elevated",
            "Installment loan balances relative to original amounts are high",
            "Installment loan balances relative to original amounts are moderate",
            "Too few accounts currently paid as agreed",
            "Length of time accounts have been established",
            "Delinquent past or present on accounts",
            "Number of accounts with delinquency",
            "Serious delinquency or public record",
            "Too few revolving accounts",
            "No bureau record or public record found",
            "No significant risk factors identified",
        ]

    ordered = []
    for p in priority:
        if p in reason_codes:
            ordered.append(p)
    for r in reason_codes:
        if r not in ordered:
            ordered.append(r)
    reason_codes = ordered[:4]

    # Clamp score for reporting
    if score < 0:
        score = 0.0
    if score > 100:
        score = 100.0

    return {"decision": decision, "reason_codes": reason_codes, "score": float(round(score, 2))}