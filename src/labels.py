# src/labels.py
from __future__ import annotations
import duckdb
from pathlib import Path

### notes on the txn.parquet and receipt.parquet:
### txn: 
    # added txn_key, Coupon_id_code reasonablely imputed, still-missing Coupon_id excluded,
    # sorted by Pay_date and txn_key.
    # (although not used here) flags.
    # no dup in rows
### receipt:
    # added receipt_key, 
    # sorted by Receive_date and receipt_key.
    # no dup in rows

def build_labels(
    receipts_parquet: Path,
    txns_parquet: Path,
    out_parquet: Path,
    short_days: int = 15,
    threads: int = 8,
) -> None:
    """
    Create receipt-level labels + audit columns.

    - label_same_user_fh: same-user redemption within [start_eff, end_eff]
    - label_same_user_st: same-user redemption within [start_eff, min(end_eff, Receive_date + short_days)]
    
    - first_valid_txn_key / _time (earliest valid txn in full horizon)
    - same_user_valid_txn_count / early / late
    - other_user_in_window_count
    - other_user_without_own_receipt_count

    Writes a parquet with one row per receipt_key.

    Assumptions (first line includes all needed fields for this script):
    - Parquets contain: receipts: (receipt_key, User_id_code, Coupon_id_code, Receive_date, Start_date, End_date,
                            Coupon_status, Coupon_amt_cent, Price_limit_cent)
                          txns:     (txn_key, User_id_code, Coupon_id_code, Pay_date,
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
                 ELSE greatest(Receive_date, Start_date)
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
        early AS (
          SELECT r.receipt_key, COUNT(*) AS same_user_early_txn_count
          FROM r
          JOIN txns t ON t.t_user=r.r_user AND t.t_coupon=r.r_coupon
          WHERE r.start_eff IS NOT NULL AND t.Pay_date < r.start_eff
          GROUP BY r.receipt_key
        ),
        late AS (
          SELECT r.receipt_key, COUNT(*) AS same_user_late_txn_count
          FROM r
          JOIN txns t ON t.t_user=r.r_user AND t.t_coupon=r.r_coupon
          WHERE r.end_eff IS NOT NULL AND t.Pay_date > r.end_eff
          GROUP BY r.receipt_key
        ),
        otheru AS (
          SELECT r.receipt_key, COUNT(*) AS other_user_in_window_count
          FROM r
          JOIN txns t ON t.t_coupon = r.r_coupon
                      AND t.t_user <> r.r_user
                      AND r.start_eff IS NOT NULL AND r.end_eff IS NOT NULL
                      AND t.Pay_date BETWEEN r.start_eff AND r.end_eff
          GROUP BY r.receipt_key
        )
        SELECT r.receipt_key,
               COALESCE(inwin.same_user_valid_txn_count, 0) AS same_user_valid_txn_count,
               COALESCE(early.same_user_early_txn_count, 0) AS same_user_early_txn_count,
               COALESCE(late.same_user_late_txn_count,  0)  AS same_user_late_txn_count,
               COALESCE(otheru.other_user_in_window_count,0) AS other_user_in_window_count
        FROM r
        LEFT JOIN inwin  USING(receipt_key)
        LEFT JOIN early  USING(receipt_key)
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
                WHERE r2.receipt_key IS NULL
                """)
    
    # table: other_user_wo_own_receipt_counts
    con.execute("""
                CREATE OR REPLACE TABLE other_user_wo_own_receipt_counts AS
                SELECT
                    receipt_key,
                    COUNT(DISTINCT txn_key) AS other_user_without_own_receipt_count
                FROM other_user_wo_own_receipt
                GROUP BY receipt_key
                """)
    

    # =====================================
    # SECTION 7: Final labels (fh, st)
    # =====================================

    # table: labels
    con.execute("""
        CREATE OR REPLACE TABLE labels AS
        SELECT
            r.receipt_key,
            r.r_user        AS User_id,
            r.r_coupon      AS Coupon_id,
            r.Receive_date, r.Start_date, r.End_date,
            r.start_eff, r.end_eff, r.short_end,
            CASE WHEN EXISTS (SELECT 1 FROM cand_fh WHERE cand_fh.receipt_key = r.receipt_key)
                 THEN 1 ELSE 0 END AS label_same_user_fh,
            CASE WHEN EXISTS (SELECT 1 FROM cand_st WHERE cand_st.receipt_key = r.receipt_key)
                 THEN 1 ELSE 0 END AS label_same_user_st
        FROM r
    """)

    # ===================================================
    # SECTION 8: Merge audits into final labels_out
    # ===================================================

    con.execute("""
        CREATE OR REPLACE TABLE labels_out AS
        SELECT
          L.*,
          M.Coupon_status,
          M.Coupon_amt_cent,
          M.Price_limit_cent,
          F.first_valid_txn_key,
          F.first_valid_txn_time,
          A.same_user_valid_txn_count,
          A.same_user_early_txn_count,
          A.same_user_late_txn_count,
          A.other_user_in_window_count,
          COALESCE(W.other_user_without_own_receipt_count, 0) AS other_user_without_own_receipt_count
        FROM labels L
        LEFT JOIN receipts_ad_fields M USING (receipt_key)
        LEFT JOIN first_valid F USING (receipt_key)
        LEFT JOIN audit_counts A USING (receipt_key)
        LEFT JOIN other_user_wo_own_receipt_counts W USING (receipt_key)
    """)

    # ==========================================
    # SECTION 9: Write parquet and close
    # ==========================================
    con.sql("SELECT * FROM labels_out").write_parquet(str(out_parquet))
    con.close()
