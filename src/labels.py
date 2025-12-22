# src/labels.py
from __future__ import annotations
import duckdb
from pathlib import Path

### notes on the txn.parquet and receipt.parquet:
### txn: 
    # added txn_key, Coupon_id_code reasonablely imputed, still-missing Coupon_id excluded,
    # masks: coupon_id_imputed, flag_no_coupon, flag_ambiguous_txn.
    # no dup in rows
### receipt:
    # added receipt_key, 
    # no dup in rows

def build_labels(
    receipts_parquet: Path,
    txns_parquet: Path,
    out_parquet: Path,          # out_parquet's name should reflect the reconcile's strictness mode: strict(1) or relax(0).
    reconcile_strict: bool = 0, # 1: not allowing txns with an imputed coupon_id; 0: allowed.
    short_days: int = 15,
    threads: int = 8,
) -> None:
    """
    Create receipt-level labels + audit columns.

    - label_same_user_fh: same-user redemption within [start_eff, end_eff]
    - label_same_user_st: same-user redemption within [start_eff, min(end_eff, Receive_date + short_days)]
    - label_valid:        no early / late / cross-user redemption c.d.t the valid fh usage window
    - first_valid_txn_key / _time (earliest valid txn in full horizon)
    - same_user_valid_txn_count / early / late
    - other_user_in_window_count
    - other_user_without_own_receipt_count

    Writes a parquet with one row per receipt_key.

    Assumptions (first line includes all needed fields for this script):
    - Parquets contain: receipts: (receipt_key, User_id_code, Coupon_id_code, Receive_date, Start_date, End_date,
                            Coupon_status, Coupon_amt_cent, Price_limit_cent)
                          txns:     (txn_key, User_id_code, Coupon_id_code, Pay_date, coupon_id_imputed,
                          Shop_id_code, Order_id_code, Coupon_type, Biz_code, Actual_pay_cent, Reduce_amount_cent)
    - Inclusive time windows (BETWEEN) per data_spec.
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # =========================
    # SECTION 1: Load & cast
    # =========================

    # table: receipts
    con.execute("""
        CREATE OR REPLACE TABLE receipts AS
        SELECT
            CAST(receipt_key AS BIGINT)            AS receipt_key,
            CAST(User_id_code     AS BIGINT)       AS r_user,
            CAST(Coupon_id_code   AS BIGINT)       AS r_coupon,
            CAST(Receive_date AS TIMESTAMP)        AS Receive_date,
            CAST(Start_date   AS TIMESTAMP)        AS Start_date,
            CAST(End_date     AS TIMESTAMP)        AS End_date
        FROM read_parquet(?)
    """, [str(receipts_parquet)])

    # table: receipts_ad_fields
    con.execute("""
                CREATE OR REPLACE TABLE receipts_ad_fields AS
                SELECT
                    CAST(receipt_key AS BIGINT)    AS receipt_key,
                    Coupon_status, Coupon_amt_cent, Price_limit_cent
                FROM read_parquet(?)
                """, [str(receipts_parquet)])

    # table: txns
    con.execute("""
            CREATE OR REPLACE TABLE txns AS
            SELECT
                CAST(txn_key    AS BIGINT)      AS txn_key,
                CAST(User_id_code    AS BIGINT) AS t_user,
                CAST(Coupon_id_code  AS BIGINT) AS t_coupon,
                CAST(Pay_date   AS TIMESTAMP)   AS Pay_date
            FROM read_parquet(?)
        """, [str(txns_parquet)])
    
    if reconcile_strict:
        con.execute("""
            CREATE OR REPLACE TABLE txns AS
            SELECT * FROM txns
            WHERE coupon_id_imputed = 0
        """, [str(txns_parquet)])
        
    
    # ===========================================
    # SECTION 2: Effective windows per receipt
    # ===========================================

    # table: r (newly added cols: start_eff, end_eff, short_end)
    con.execute(f"""
        CREATE OR REPLACE TABLE r AS
        SELECT
            receipt_key, r_user, r_coupon, Receive_date, Start_date, End_date,
            CASE WHEN Start_date IS NULL OR Receive_date IS NULL
                 THEN NULL
                 ELSE GREATEST(Receive_date, Start_date)
            END AS start_eff,
            End_date AS end_eff,
            CASE
                WHEN End_date IS NULL OR Receive_date IS NULL THEN NULL
                ELSE LEAST(End_date, Receive_date + INTERVAL {short_days} DAY)
            END AS short_end
        FROM receipts
    """)

    # ==================================================
    # SECTION 3: Candidate same-user matches by window
    # ==================================================

    # table: cand_fh
    con.execute("""
        CREATE OR REPLACE TABLE cand_fh AS
        SELECT r.receipt_key, t.txn_key, t.Pay_date
        FROM r
        JOIN txns t
          ON t.t_user = r.r_user
         AND t.t_coupon = r.r_coupon
         AND r.start_eff IS NOT NULL AND r.end_eff IS NOT NULL
         AND t.Pay_date BETWEEN r.start_eff AND r.end_eff
    """)

    # table: cand_st
    con.execute("""
        CREATE OR REPLACE TABLE cand_st AS
        SELECT r.receipt_key, t.txn_key, t.Pay_date
        FROM r
        JOIN txns t
          ON t.t_user = r.r_user
         AND t.t_coupon = r.r_coupon
         AND r.start_eff IS NOT NULL AND r.short_end IS NOT NULL
         AND t.Pay_date BETWEEN r.start_eff AND r.short_end
    """)

    # =====================================
    # SECTION 4: Earliest valid txn (audit)
    # =====================================

    # table: first_valid
    ## this is not necessary as we are guessing what is the truth now:
    con.execute("""
        CREATE OR REPLACE TABLE first_valid AS
        SELECT receipt_key,
               MIN_BY(txn_key, Pay_date) AS first_valid_txn_key,
               MIN(Pay_date)             AS first_valid_txn_time
        FROM cand_fh
        GROUP BY receipt_key
    """)

    # ======================================
    # SECTION 5: Audit_counts part 1: 
    # Same-user early/late counts
    # ======================================

    # table: audit_counts
    con.execute("""
        CREATE OR REPLACE TABLE audit_counts AS
        WITH inwin AS (
          SELECT receipt_key, COUNT(*) AS same_user_valid_txn_count
          FROM cand_fh GROUP BY receipt_key
        ),
                
        --- assume the coupons with the same coupon_id have the same start & end time
        --- one txn might be matched to multiple receipt events with the same coupon_id
        --- for each receipt event, if the pay date is before the coupon start date, it is always invalid.
                      
        early_1 AS (
          SELECT r.receipt_key, COUNT(*) AS same_user_early_txn_count_1
          FROM r
          JOIN txns t ON t.t_user=r.r_user AND t.t_coupon=r.r_coupon
          WHERE r.Start_date IS NOT NULL AND t.Pay_date < r.Start_date
          GROUP BY r.receipt_key
        ),
        
        --- multiple txns might be matched to the same receipt event (defined by receipt_key)
        --- if the lastest txn happens before the receive date, the receipt event is invalid.
        --- the gist behind is when receive date is later than start date, there should be at least one txn happens after the receive date.       
        
        early_2 AS (
          SELECT r.receipt_key, COUNT(*) AS same_user_early_txn_count_2
          FROM r
          JOIN txns t 
                ON t.t_user=r.r_user AND t.t_coupon=r.r_coupon
                AND r.Start_date IS NOT NULL AND r.end_eff IS NOT NULL AND r.Receive_date IS NOT NULL
                AND t.Pay_date BETWEEN r.Start_date AND r.end_eff
          GROUP BY r.receipt_key
          HAVING MAX(t.Pay_date) < r.Receive_date  
        ),
                
        late AS (
          SELECT r.receipt_key, COUNT(*) AS same_user_late_txn_count
          FROM r
          JOIN txns t ON t.t_user=r.r_user AND t.t_coupon=r.r_coupon
          WHERE r.end_eff IS NOT NULL AND t.Pay_date > r.end_eff
          GROUP BY r.receipt_key
        ),
                
        --- for each receipt event, find the txns of other users using the same coupon (same id) within the valid usage window
       
        otheru AS (
          SELECT r.receipt_key, COUNT(*) AS other_user_in_window_txn_count
          FROM r
          JOIN txns t ON t.t_coupon = r.r_coupon
                      AND t.t_user <> r.r_user 
                      AND r.start_eff IS NOT NULL AND r.end_eff IS NOT NULL
                      AND t.Pay_date BETWEEN r.start_eff AND r.end_eff
          GROUP BY r.receipt_key
        )
    
        SELECT r.receipt_key,
               COALESCE(inwin.same_user_valid_txn_count, 0) AS same_user_valid_txn_count,
               COALESCE(early_1.same_user_early_txn_count_1, 0) AS same_user_early_txn_count_1,
               COALESCE(early_2.same_user_early_txn_count_2, 0) AS same_user_early_txn_count_2,
               COALESCE(late.same_user_late_txn_count,  0)  AS same_user_late_txn_count,
               COALESCE(otheru.other_user_in_window_txn_count,0) AS other_user_in_window_txn_count
        FROM r
        LEFT JOIN inwin  USING(receipt_key)
        LEFT JOIN early_1 USING(receipt_key)
        LEFT JOIN early_2 USING(receipt_key)
        LEFT JOIN late   USING(receipt_key)
        LEFT JOIN otheru USING(receipt_key)
    """)

    # ==============================================================
    # SECTION 6: Audit_counts part 2: 
    # Other-user WITHOUT OWN RECEIPT covering the txn
    # ==============================================================

    con.execute("""
            CREATE OR REPLACE TABLE other_user_hits AS
            SELECT
                r.receipt_key,
                r.r_coupon             AS coupon_id,
                t.t_user               AS other_user,
                t.txn_key,
                t.Pay_date
            FROM r
            JOIN txns t
                ON t.t_coupon = r.r_coupon
                AND t.t_user <> r.r_user
                AND r.start_eff IS NOT NULL AND r.end_eff IS NOT NULL
                AND t.Pay_date BETWEEN r.start_eff AND r.end_eff
                """)
    
    con.execute("""
                CREATE OR REPLACE TABLE other_user_wo_own_receipt AS
                SELECT
                    o.receipt_key,
                    o.txn_key
                FROM other_user_hits o
                LEFT JOIN r r2
                    ON r2.r_user = o.other_user
                    AND r2.r_coupon = o.coupon_id
                    AND r2.start_eff IS NOT NULL AND r2.end_eff IS NOT NULL
                    AND o.Pay_date BETWEEN r2.start_eff AND r2.end_eff
                WHERE r2.receipt_key IS NULL   --- the key part: no receipt event is matched
                """)
    
    # table: other_user_wo_own_receipt_txn_counts
    con.execute("""
                CREATE OR REPLACE TABLE other_user_wo_own_receipt_counts AS
                SELECT
                    receipt_key,
                    COUNT(DISTINCT txn_key) AS other_user_without_own_receipt_txn_count
                FROM other_user_wo_own_receipt
                GROUP BY receipt_key
                """)
    

    # ==========================================
    # SECTION 7: Final labels (fh, st)
    # Also generate the flags related to invalid redemption
    # ==========================================

    # table: labels
    con.execute("""
        CREATE OR REPLACE TABLE labels AS
        SELECT
            r.receipt_key,
            r.r_user        AS User_id_code,
            r.r_coupon      AS Coupon_id_code,
            r.Receive_date, r.Start_date, r.End_date,
            r.start_eff, r.end_eff, r.short_end,
            CASE WHEN EXISTS (SELECT 1 FROM cand_fh WHERE cand_fh.receipt_key = r.receipt_key)
                 THEN 1 ELSE 0 END AS label_same_user_fh,
            CASE WHEN EXISTS (SELECT 1 FROM cand_st WHERE cand_st.receipt_key = r.receipt_key)
                 THEN 1 ELSE 0 END AS label_same_user_st
        FROM r
    """)

    # generate flag_early and flag_late
    con.execute("""
        CREATE OR REPLACE TABLE flags_early_late_redeem AS
                SELECT
                    A.receipt_key,
                    A.same_user_valid_txn_count,
                    A.same_user_early_txn_count_1,
                    A.same_user_early_txn_count_2,
                    (A.same_user_early_txn_count_1 + A.same_user_early_txn_count_2) AS same_user_early_txn_count,
                    A.same_user_late_txn_count,
                    A.other_user_in_window_txn_count,
                    COALESCE(W.other_user_without_own_receipt_txn_count, 0) AS other_user_without_own_receipt_txn_count,
                    CASE WHEN A.same_user_early_txn_count_1 + A.same_user_early_txn_count_2 > 0
                        THEN 1 ELSE 0 END AS flag_early,
                    CASE WHEN A.same_user_late_txn_count > 0
                        THEN 1 ELSE 0 END AS flag_late,
                FROM audit_counts A
                LEFT JOIN other_user_wo_own_receipt_counts W USING (receipt_key)
    """)

    # generate flag_cross_user
    con.execute("""
        CREATE OR REPLACE TABLE flags_redeem_2 AS
                SELECT
                    f.*,
                    CASE WHEN f.other_user_without_own_receipt_txn_count > 0
                        THEN 1 ELSE 0 END AS flag_cross_user
                FROM flags_early_late_redeem f
    """)

    # ==============================================================
    # SECTION 8: Merge labels, audits, flags into final labels_out
    # ==============================================================

    # generate flag_struc_invalid
    con.execute("""
        CREATE OR REPLACE TABLE labels_out_it AS
        SELECT
          L.*,
          M.Coupon_status,
          M.Coupon_amt_cent,
          M.Price_limit_cent,
          F.first_valid_txn_key,
          F.first_valid_txn_time,
          FR.same_user_valid_txn_count,
          FR.same_user_early_txn_count,
          FR.same_user_late_txn_count,
          FR.other_user_in_window_txn_count,
          FR.other_user_without_own_receipt_txn_count, 
          FR.flag_early,
          FR.flag_late,
          FR.flag_cross_user,
          CASE WHEN ((L.User_id_code IS NULL OR L.User_id_code = -1) OR
                     (L.Coupon_id_code IS NULL OR  L.Coupon_id_code = -1) OR
                     L.Start_date IS NULL OR L.Receive_date IS NULL OR ---any missing fields lead to struc_invalid flag being 1. 
                     L.End_date IS NULL OR L.Start_date > L.End_date)
                THEN 1 ELSE 0 END AS flag_struc_invalid
        FROM labels L
        LEFT JOIN receipts_ad_fields M USING (receipt_key)
        LEFT JOIN first_valid F USING (receipt_key)
        LEFT JOIN flags_redeem_2 FR USING (receipt_key)
    """)

    # generate label_valid
    con.execute("""
        CREATE OR REPLACE TABLE labels_out AS
                SELECT 
                    L.*,
                    CASE WHEN COALESCE(L.flag_early, 0) <> 0
                            OR COALESCE(L.flag_late, 0)  <> 0
                            OR COALESCE(L.flag_cross_user, 0) <> 0
                            OR COALESCE(L.flag_struc_invalid, 0) <> 0
                    THEN 0 ELSE 1 END AS label_valid
                FROM labels_out_it L
    """)

    # ==========================================
    # SECTION 9: Write parquet and close
    # ==========================================
    con.sql("SELECT * FROM labels_out").write_parquet(str(out_parquet))
    con.close()
