# src/flags.py
from __future__ import annotations
import duckdb
from pathlib import Path

def add_txn_level_flags(
    receipts_parquet: Path,
    txns_parquet: Path,
    txn_out_parquet: Path,  # the name should reflect the reconcile_strict status- strict or relax.
    reconcile_strict: bool = 0, # 1 means not allowing reconciliation, 0 means allowing.
    threads: int = 8,
):
    """
    Inputs:
    - txns_parquet: the reconciled/non-reconciled txn table. (switched by reconcile_strict)
    - receipts_parquet: the receipt table with the receipt_key added.

    Add the following flags to the txn table:
    - `flag_untracked_coupon`
    - `flag_missing_info`
    - `flag_pay_or_reduce_amt_abn`
    (For def. details see data_spec.md)
    
    Output:
    the txn table with added flags. In the parquet format.
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # =========================
    # SECTION 1: Load & cast
    # =========================
    # table: txns
    con.execute("""
        CREATE OR REPLACE TABLE txns AS
        SELECT
            CAST(txn_key    AS BIGINT)      AS txn_key,
            CAST(User_id_code    AS BIGINT) AS t_user,
            CAST(Coupon_id_code  AS BIGINT) AS t_coupon,
            CAST(Shop_id_code AS BIGINT)    AS Shop_id_code,
            CAST(Order_id_code AS BIGINT)   AS Order_id_code,
            CAST(Coupon_type AS BIGINT)     AS Coupon_type,
            Biz_code,
            CAST(Pay_date   AS TIMESTAMP)   AS Pay_date,
            CAST(Actual_pay_cent  AS BIGINT) AS Actual_pay_cent,
            CAST(Reduce_amount_cent AS BIGINT)  AS Reduce_amount_cent,
            CAST(coupon_id_imputed AS TINYINT)  AS coupon_id_imputed,
            CAST(flag_no_coupon AS TINYINT)     AS flag_no_coupon,
            CAST(flag_ambiguous_txn AS TINYINT)    AS flag_ambiguous_txn
        FROM read_parquet(?)
        """, [str(txns_parquet)])
    
    if reconcile_strict:
        con.execute("""
        CREATE OR REPLACE TABLE txns AS
        SELECT t.*
        FROM txns t
        WHERE coupon_id_imputed = 0
        """, [str(txns_parquet)])
    
    # table: receipts
    con.execute("""
        CREATE OR REPLACE TABLE receipts AS
        SELECT
            CAST(receipt_key AS BIGINT)            AS receipt_key,
            CAST(User_id_code     AS BIGINT)       AS r_user,
            CAST(Coupon_id_code   AS BIGINT)       AS r_coupon,
        FROM read_parquet(?)
    """, [str(receipts_parquet)])

    # =========================
    # SECTION 2.1: 
    # flag1: untracked coupon
    # =========================
    con.execute("""
        -- txns whose coupon was never received by anyone
        CREATE OR REPLACE TABLE txn_untracked AS
        SELECT t.txn_key
        FROM txns t
        LEFT ANTI JOIN receipts r
        ON t.t_coupon = r.r_coupon
        WHERE (t.t_coupon <> -1 AND t.t_coupon IS NOT NULL)
    """)

    con.execute("""
        -- attach the flag back onto txns
        CREATE OR REPLACE TABLE txns_flagged AS
        SELECT
            t.*,
        CASE WHEN u.txn_key IS NOT NULL THEN 1 ELSE 0 END::TINYINT AS flag_untracked_coupon
        FROM txns t
        LEFT JOIN txn_untracked u USING (txn_key)
    """)

    # =========================
    # SECTION 2.2: 
    # flag2: missing info
    # =========================
    con.execute("""
        -- txns whose info is incomplete
        CREATE OR REPLACE TABLE txn_info_incomplete AS
        SELECT t.txn_key
        FROM txns t
        WHERE
                (t.t_user = -1 OR t.t_user IS NULL)
                OR ((t.t_coupon = -1 OR t.t_coupon IS NULL) AND t.flag_no_coupon <> 1)
                OR (t.Shop_id_code = -1 OR t.Shop_id_code IS NULL)
                OR (t.Order_id_code = -1 OR t.Order_id_code IS NULL)
                OR t.Coupon_type IS NULL
                OR t.Biz_code IS NULL
                OR t.Pay_date IS NULL
                OR t.Actual_pay_cent IS NULL
                OR t.Reduce_amount_cent IS NULL
    """)

    con.execute("""
        -- attach the flag back onto txns
        CREATE OR REPLACE TABLE txns_flagged_2 AS
        SELECT
            t.*,
        CASE WHEN ii.txn_key IS NOT NULL THEN 1 ELSE 0 END::TINYINT AS flag_missing_info
        FROM txns_flagged t
        LEFT JOIN txn_info_incomplete ii USING (txn_key)
    """)

    # =========================
    # SECTION 2.3: 
    # flag3: abnormal payment amount or reduce amount
    # =========================
    con.execute("""
        -- txns whose payment/reduce amount is abnormal
        CREATE OR REPLACE TABLE txn_pay_or_reduce_abn AS
        SELECT t.txn_key
        FROM txns t
        WHERE
                (t.Reduce_amount_cent > 0 AND t.flag_no_coupon = 1)
                OR t.Actual_pay_cent < 0
                OR t.Reduce_amount_cent < 0
    """)

    con.execute("""
        -- attach the flag back onto txns
        CREATE OR REPLACE TABLE txns_flagged_3 AS
        SELECT
            t.*,
        CASE WHEN ab.txn_key IS NOT NULL THEN 1 ELSE 0 END::TINYINT AS flag_pay_or_reduce_amt_abn
        FROM txns_flagged_2 t
        LEFT JOIN txn_pay_or_reduce_abn ab USING (txn_key)
    """)

    # ==========================================
    # SECTION 3: Write parquet and close
    # ==========================================

    # cast the t_user, t_coupon fields names back to User_id_code, Coupon_id_code

    con.execute("""
        ALTER TABLE txns_flagged_3 
        RENAME COLUMN t_user TO User_id_code
        RENAME COLUMN t_coupon TO Coupon_id_code;
    """)

    con.sql("SELECT * FROM txns_flagged_3").write_parquet(str(txn_out_parquet))
    con.close()

