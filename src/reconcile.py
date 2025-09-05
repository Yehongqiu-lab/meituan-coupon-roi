# src/reconcile.py
# note: before running this, please run clean_normalize.py to clean and normalize the raw data.

import pandas as pd

# public
def impute_missing_coupon_ids(txns_df, receipts_df):
    """
    Step-1 reconciliation: for txns with missing Coupon_id,
    impute when exactly one same-user receipt is valid at Pay_date.
    Otherwise set flags: flag_no_coupon vs flag_ambiguous_txn.
    Returns txns_df with columns updated/added.
    """

    # add stable keys
    txns_df, receipts_df = _add_keys(txns_df, receipts_df)

    # filter to txns with missing Coupon_id_code
    txn_missing_mask = txns_df["Coupon_id_code"] == -1
    txn_missing_df = txns_df[txn_missing_mask]
    print(f"[txn] {len(txn_missing_df)} rows with missing Coupon_id_code out of {len(txns_df)} total rows.")

    if len(txn_missing_df) == 0:
        # nothing to do
        txns_df["flag_no_coupon"] = 0
        txns_df["flag_ambiguous_txn"] = 0
        txns_df["coupon_id_imputed"] = 0
        return txns_df

    # prep receipts for matching
    rec_prepped_df = _prep_receipts_for_matching(receipts_df)

    # build candidate matches by (User_id) join + window filter
    candidates_df, txn_imputable, txn_pre_ambig = _candidate_matches(txn_missing_df, rec_prepped_df)

    # decide per txn_key
    decisions_df = _decide_per_txn(candidates_df, txn_imputable)

    # apply decisions back to txns_df
    txns_df = _apply_decisions(txns_df, decisions_df, txn_pre_ambig)

    return txns_df

#################################################################################################
### internal helpers for the public function `impute_missing_coupon_ids(txns_df, receipts_df)`

def _add_keys(txns_df: pd.DataFrame, receipts_df: pd.DataFrame):
    """
    Add stable keys to transaction and receipt tables before reconciliation.

    - txn_key: original index of txn_df (after cleaning/deduplication).
    - receipt_key: original index of receipts_df.
    """

    # Defensive copy to avoid mutating caller’s frames
    txns_df = txns_df.copy()
    receipts_df = receipts_df.copy()

    # Ensure indexes are unique and stable
    if not txns_df.index.is_unique:
        raise ValueError("Transaction DataFrame index is not unique — check deduplication in clean_normalize.py")
    if not receipts_df.index.is_unique:
        raise ValueError("Receipt DataFrame index is not unique — check deduplication in clean_normalize.py")

    txns_df["txn_key"] = txns_df.index
    receipts_df["receipt_key"] = receipts_df.index

    return txns_df, receipts_df


def _prep_receipts_for_matching(receipts_df):
    """Add start_eff := max(Receive_date, Start_date). 
    Drop rows with missing dates or Start_date > End_date."""

    df = receipts_df.copy()

    df = df.drop(columns=["Price_limit_cent", "Coupon_status", "Coupon_amt_cent"])  # not needed for matching
    
    # drop rows with User_id_code == -1
    mask = df["User_id_code"] == -1
    df = df.drop(df[mask].index) 

    # label and drop invalid coupons
    df = _label_receipts_with_missing_valid_usage_window(receipts_df)
    df = df[df["flag_invalid_coupon"] == 0]

    # if empty after filtering, return empty with needed columns
    if len(df) == 0:
        df["start_eff"] = pd.NaT
        df = df.drop(columns=["flag_invalid_coupon", "Receive_date", "Start_date"])
        return df
    
    # effective start date is max(Receive_date, Start_date)
    df["start_eff"] = df[["Receive_date", "Start_date"]].max(axis=1)

    # drop unneeded columns
    df = df.drop(columns=["flag_invalid_coupon", "Receive_date", "Start_date"])

    return df

def _candidate_matches(txn_missing_df, rec_prepped_df):
    """
    Build candidate matches by (User_id) join + window filter:
    keep rows where start_eff ≤ Pay_date ≤ End_date.
    Returns:
    1) a dataframe of candidates with:
      ['txn_key', 'User_id_code','Pay_date','Reduce_amount_cent',
       'receipt_key','Coupon_id_code','start_eff','End_date']
    2) txn_imputable: the subset of txn_missing_df with non-missing User_id and Pay_date (for later decision).
    3) txn_pre_ambig: the subset of txn_missing_df with missing User_id or Pay_date (treated as ambiguous txn).
    """
    
    txn_missing_df = txn_missing_df.copy()
    rec_prepped_df = rec_prepped_df.copy()

    # pre-label ambiguous txns with missing User_id or Pay_date
    pre_ambig_mask = ((txn_missing_df['User_id_code'] == -1) 
                      | txn_missing_df['Pay_date'].isna())
    cols = ["txn_key", "User_id_code", "Pay_date", "Reduce_amount_cent"]
    txn_pre_ambig = txn_missing_df.loc[pre_ambig_mask, cols].copy()   # for report and later usage
    if len(txn_pre_ambig) > 0:
        print(f"[txn] pre-labeled {len(txn_pre_ambig)} ambiguous txns with missing User_id or Pay_date.")
    assert txn_pre_ambig["txn_key"].is_unique, "Internal error: txn_key not unique in pre-ambiguous txns"

    # filter to rows with User_id and Pay_date
    txn_imputable = txn_missing_df.loc[~pre_ambig_mask, cols].copy()  # go to join + window filter
    print(f"[txn] {len(txn_imputable)} rows with non-missing User_id and Pay_date for potential imputation.")
    assert txn_imputable["txn_key"].is_unique, "Internal error: txn_key not unique in imputable txns"

    # join
    candidates_df = pd.merge(txn_imputable, rec_prepped_df, on="User_id_code", how="inner")

    # window filter
    cond = (candidates_df["start_eff"] <= candidates_df["Pay_date"]) & \
           (candidates_df["Pay_date"] <= candidates_df["End_date"])
    candidates_df = candidates_df[cond]
    return candidates_df, txn_imputable, txn_pre_ambig


def _decide_per_txn(candidates_df, txn_imputable):
    """
    Collapse candidates to one row per txn_key with decision fields:
      - match_count
      - receipt_key (if match_count == 1)
      - Coupon_id_code   (if match_count == 1)
      - coupon_id_imputed, flag_no_coupon, flag_ambiguous_txn
    """
    
    cd_df = candidates_df.copy()

    # count matches per txn_key
    # summary per txn_key (one row per group)
    summary = (
        cd_df.groupby('txn_key', as_index=False)
             .agg(match_count=('txn_key', 'size'),
                Coupon_id_code=('Coupon_id_code', 'first'),
                receipt_key=('receipt_key', 'first'),
    ))

    # defensive: ensure txn_key is unique in summary
    assert summary["txn_key"].is_unique, "Internal error: txn_key not unique in summary"

    # merge with txn_imputable to get all txn_keys
    decisions_df = (txn_imputable[["txn_key", "Reduce_amount_cent"]]
                    .merge(summary, on="txn_key", how="left")
                    .fillna({"match_count": 0, "Coupon_id_code": -1, "receipt_key": -1}))
    assert decisions_df["txn_key"].is_unique, "Internal error: txn_key not unique in decisions_df"
    
    # set flags
    decisions_df["coupon_id_imputed"] = 0
    decisions_df["flag_no_coupon"] = 0
    decisions_df["flag_ambiguous_txn"] = 0

    decisions_df.loc[decisions_df["match_count"] == 1, "coupon_id_imputed"] = 1 # imputed
    # no coupon if no match and Reduce_amount_cent < 1 cent
    decisions_df.loc[((decisions_df["match_count"] == 0) 
                      & (abs(decisions_df["Reduce_amount_cent"]) < 1)), "flag_no_coupon"] = 1
    # ambiguous otherwise
    decisions_df.loc[((decisions_df["coupon_id_imputed"] == 0) 
                      & (decisions_df["flag_no_coupon"] == 0)), "flag_ambiguous_txn"] = 1
    
    return decisions_df

    
def _apply_decisions(txns_df, decisions_df, txn_pre_ambig):
    """Write imputed Coupon_id and flags back to txns_df; 
    Write pre-filtered ambiguous txns back to txns_df;
    fill missing flags with 0."""
    
    txns_df = txns_df.copy()
    
    # merge decisions
    txns_df = txns_df.merge(decisions_df[["txn_key",
                                           "flag_no_coupon", "flag_ambiguous_txn", "coupon_id_imputed"]],
                            on="txn_key", how="left")
    assert txns_df["txn_key"].is_unique, "Internal error: txn_key not unique after merge with decisions"

    # write imputed Coupon_id_code
    s = (decisions_df.query("coupon_id_imputed == 1")
            .set_index("txn_key")["Coupon_id_code"])
    mask = txns_df["coupon_id_imputed"] == 1
    txns_df.loc[mask, "Coupon_id_code"] = txns_df.loc[mask, "txn_key"].map(s)

    # write pre-labeled ambiguous txns
    mask = txns_df["txn_key"].isin(txn_pre_ambig["txn_key"])
    txns_df.loc[mask, "flag_ambiguous_txn"] = 1
    
    # fill missing flags with 0
    flags = ["flag_no_coupon", "flag_ambiguous_txn", "coupon_id_imputed"]
    txns_df[flags] = txns_df[flags].fillna(0).astype("Int8")
    
    return txns_df


##################################################################################################
### internal helpers for `flag_invalid_coupon`
def _label_receipts_with_missing_valid_usage_window(df):
    """Label receipts with missing or invalid usage window
    with the flag `flag_invalid_coupon`."""
    df = df.copy()
    n_before = len(df)
    cond = (df["Start_date"].isna() | df["End_date"].isna() | 
            (df["Start_date"] > df["End_date"]) | df["Receive_date"].isna())
    df["flag_invalid_coupon"] = 0
    df.loc[cond, "flag_invalid_coupon"] = 1
    n_after = df["flag_invalid_coupon"].sum()
    if n_before != n_after:
        print(f"[receipt] labeled {n_after} invalid coupons (missing/invalid usage window \
              or missing receive date) out of {n_before} rows.")
    return df