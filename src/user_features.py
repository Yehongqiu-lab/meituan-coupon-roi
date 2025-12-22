# src/user_features.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd

## user features
    # the HISTORICAL invalidity rate for this user on all their previously received coupons
    # the HISTORICAL redemption rate for this user on all their previously received coupons
    # the exceeding part of the user's HISTORICAL average spend compared to THIS coupon's price limit
    # the user's HISTORICAL frequency of visits
    # the user's HISTORICAL frequency of purchases
def user_features(
        receipts_labelled_parquet: Path,
        txns_parquet: Path,
        visits_parquet: Path,
        out_parquet: Path,
        lookback_days: list[int] = [8, 15, 31],
        threads: int = 8
) -> None:
    """
    Generate user features and save to out_parquet."""
    
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # ===================
    # Section 1: load
    # ===================
    
    # load coupon receipts
    con.execute("""
        CREATE TABLE receipts AS
            SELECT
                receipt_key,
                User_id_code,
                Coupon_id_code,
                Price_limit_cent,
                Coupon_amt_cent,
                Receive_date, 
                CASE WHEN label_valid = 1 THEN 0 ELSE 1 END AS label_invalid,
                label_same_user_fh,
                label_same_user_st
            FROM read_parquet(?)""", [str(receipts_labelled_parquet)])
    
    # load txns
    con.execute("""
        CREATE TABLE txns AS
            SELECT
                User_id_code,
                Order_id_code,
                Pay_date,
                Actual_pay_cent, Reduce_amount_cent
            FROM read_parquet(?)""", [str(txns_parquet)])
    
    # load visits
    con.execute("""
        CREATE TABLE visits AS
            SELECT
                User_id_code,
                CAST (Visit_date AS TIMESTAMP) AS Visit_date    
            FROM read_parquet(?)
    """, [str(visits_parquet)])

    # ===================================
    # Section 2.1: HISTORICAL redemption 
    # & invalidity rate for this user 
    # of all their received coupons
    # ===================================

    con.execute("""
        CREATE TABLE cum AS
            -- pre-aggregate to daily
            WITH daily AS (
                SELECT
                    User_id_code, Receive_date,
                    SUM(label_invalid)      AS invalid_daily,
                    SUM(label_same_user_fh) AS fh_redeem_daily,
                    SUM(label_same_user_st) AS st_redeem_daily,
                    COUNT(*)                AS vol_daily
                FROM receipts
                GROUP BY 1,2
            )
            SELECT
                User_id_code, Receive_date,
                SUM(invalid_daily)   OVER same_user AS invalid_cum,
                SUM(fh_redeem_daily) OVER same_user AS fh_redeem_cum,
                SUM(st_redeem_daily) OVER same_user AS st_redeem_cum,
                SUM(vol_daily)       OVER same_user AS vol_cum
            FROM daily
            WINDOW same_user AS (
                PARTITION BY User_id_code
                ORDER BY Receive_date  ---this will put the lines with NaT receive_date to the end.
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )     
    """)
    con.execute("""
        CREATE TABLE asof_right AS
            SELECT 
                r.receipt_key,
                (c1.vol_cum IS NULL)          AS no_history_indicator,
                COALESCE(c1.invalid_cum, 0)   AS invalid_right,
                COALESCE(c1.fh_redeem_cum, 0) AS fh_redeem_right,
                COALESCE(c1.st_redeem_cum, 0) AS st_redeem_right,
                COALESCE(c1.vol_cum, 0)       AS vol_right
            FROM receipts r
            ASOF LEFT JOIN cum c1
                ON r.User_id_code = c1.User_id_code
                AND (r.Receive_date - INTERVAL 1 DAY) >= c1.Receive_date            
    """)
    for window_len in lookback_days:
        con.execute(f"""
            CREATE OR REPLACE TABLE rlookback AS
                WITH asof_left AS (
                    SELECT
                        r.receipt_key,
                        (c2.vol_cum IS NULL)          AS no_history_indicator,
                        COALESCE(c2.invalid_cum, 0)   AS invalid_left,
                        COALESCE(c2.fh_redeem_cum, 0) AS fh_redeem_left,
                        COALESCE(c2.st_redeem_cum, 0) AS st_redeem_left,
                        COALESCE(c2.vol_cum, 0)       AS vol_left
                    FROM receipts r
                    ASOF LEFT JOIN cum c2
                        ON r.User_id_code = c2.User_id_code
                        AND (r.Receive_date - INTERVAL {window_len} DAY) >= c2.Receive_date
                )
                SELECT
                    a.receipt_key,
                    (a.no_history_indicator OR b.no_history_indicator)::INT AS no_history_indicator,
                    (a.invalid_right - b.invalid_left)                      AS invalid_{window_len-1}d,
                    (a.fh_redeem_right - b.fh_redeem_left)                  AS fh_redeem_{window_len-1}d,
                    (a.st_redeem_right - b.st_redeem_left)                  AS st_redeem_{window_len-1}d,
                    (a.vol_right - b.vol_left)                              As vol_{window_len-1}d
                FROM asof_right a
                LEFT JOIN asof_left b
                    ON a.receipt_key = b.receipt_key
        """)

        con.execute(f"""
            CREATE OR REPLACE TABLE receipts AS
                SELECT r.*,
                    bk.no_history_indicator AS no_hist_rcs_marker_{window_len-1}d,
                    CASE
                        WHEN bk.vol_{window_len-1}d = 0 THEN 0.000
                        ELSE ROUND(bk.invalid_{window_len-1}d / bk.vol_{window_len-1}d, 3)
                        END AS Rate_same_user_invalid_{window_len-1}d,
                    CASE
                        WHEN bk.vol_{window_len-1}d = 0 THEN 0.000
                        ELSE ROUND(bk.fh_redeem_{window_len-1}d / bk.vol_{window_len-1}d, 3)
                        END AS Rate_same_user_fh_redeem_{window_len-1}d,
                    CASE
                        WHEN bk.vol_{window_len-1}d = 0 THEN 0.000
                        ELSE ROUND(bk.st_redeem_{window_len-1}d / bk.vol_{window_len-1}d, 3)
                        END AS Rate_same_user_st_redeem_{window_len-1}d
                FROM receipts r
                LEFT JOIN rlookback bk
                    ON r.receipt_key = bk.receipt_key
        """)
    
    # ======================================
    # Section 2.2: 
    # (1) HISTORICAL average spend of the user
    # v.s the coupon's price limit (Ratio);
    # (2) HISTORICAL average reduce_amt the user 
    # enjoyed v.s the coupon's amt (Ratio);
    # (3) HISTORICAL frequency of purchase
    # of the user. 
    # ======================================
    con.execute("""
        CREATE OR REPLACE TABLE cum AS
            WITH per_order AS (
                SELECT
                    User_id_code, Pay_date, Order_id_code,
                    FIRST(Actual_pay_cent)        AS Actual_pay_perorder, --- actual pay amount is the same across observations for each txn record involving different coupons
                    SUM(Reduce_amount_cent)       AS Reduce_amt_perorder
                FROM txns
                GROUP BY 1,2,3
            ),
            daily AS (
                SELECT
                    User_id_code, Pay_date,
                    COUNT(Order_id_code)          AS order_vol_daily,
                    SUM(Actual_pay_perorder)      AS txn_amt_daily,
                    SUM(Reduce_amt_perorder)      AS reduce_amt_daily
                FROM per_order
                GROUP BY 1,2
            )
            SELECT
                User_id_code, Pay_date,
                SUM(order_vol_daily)  OVER same_user AS order_vol_cum,
                SUM(txn_amt_daily)    OVER same_user AS txn_amt_cum,
                SUM(reduce_amt_daily) OVER same_user AS reduce_amt_cum
            FROM daily
            WINDOW same_user AS (
                PARTITION BY User_id_code
                ORDER BY Pay_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
    """)
    con.execute("""
        CREATE OR REPLACE TABLE asof_right AS
            SELECT r.receipt_key,
                (c1.order_vol_cum IS NULL)     AS no_hist_marker,
                COALESCE(c1.order_vol_cum, 0)  AS order_vol_right,
                COALESCE(c1.txn_amt_cum, 0)    AS txn_amt_right,
                COALESCE(c1.reduce_amt_cum, 0) AS reduce_amt_right
            FROM receipts r
            ASOF LEFT JOIN cum c1
                ON r.User_id_code = c1.User_id_code
                AND (r.Receive_date - INTERVAL 1 DAY) >= c1.Pay_date
    """)
    for window_len in lookback_days:
        con.execute(f"""
            CREATE OR REPLACE TABLE asof_left AS
                SELECT r.receipt_key,
                    (c2.order_vol_cum IS NULL)     AS no_hist_marker,
                    COALESCE(c2.order_vol_cum, 0)  AS order_vol_left,
                    COALESCE(c2.txn_amt_cum, 0)    AS txn_amt_left,
                    COALESCE(c2.reduce_amt_cum, 0) AS reduce_amt_left
                FROM receipts r
                ASOF LEFT JOIN cum c2
                    ON r.User_id_code = c2.User_id_code
                    AND (r.Receive_date - INTERVAL {window_len} DAY) >= c2.Pay_date
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE rlookback AS
                SELECT a.receipt_key,
                    (a.no_hist_marker OR b.no_hist_marker)::INT AS no_hist_marker,
                    (a.order_vol_right - b.order_vol_left)      AS order_vol_{window_len-1}d,
                    (a.txn_amt_right - b.txn_amt_left)          AS txn_amt_{window_len-1}d,
                    (a.reduce_amt_right - b.reduce_amt_left)    AS reduce_amt_{window_len-1}d
                FROM asof_right a
                LEFT JOIN asof_left b
                    ON a.receipt_key = b.receipt_key
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE receipts AS
                SELECT r.*,
                    bk.no_hist_marker AS no_hist_txns_marker_{window_len-1}d,

                    ---avgspend does not include reduced amt here.
                    CASE
                        WHEN bk.order_vol_{window_len-1}d = 0 THEN 0.000
                        ELSE ROUND(bk.txn_amt_{window_len-1}d / bk.order_vol_{window_len-1}d / (r.Price_limit_cent + 1), 3) 
                        END AS Rt_avgspend_vs_pricelimit_{window_len-1}d,
                    
                    CASE
                        WHEN bk.order_vol_{window_len-1}d = 0 THEN 0.000
                        ELSE ROUND(bk.reduce_amt_{window_len-1}d / bk.order_vol_{window_len-1}d / (r.Coupon_amt_cent + 1), 3)
                        END AS Rt_avgreduce_vs_couponamt_{window_len-1}d,
                    ROUND(bk.order_vol_{window_len-1}d / {window_len-1}, 3)
                        AS Freq_purchase_{window_len-1}d
                FROM receipts r
                LEFT JOIN rlookback bk
                    ON r.receipt_key = bk.receipt_key
        """)
    
    # ===========================================
    # Section 2.3: 
    # each user's HISTORICAL frequency of visit
    # ===========================================
    con.execute("""
        CREATE OR REPLACE TABLE cum AS
            WITH daily AS (
                SELECT User_id_code, Visit_date,
                    COUNT(*) AS visit_tms_daily
                FROM visits
                GROUP BY 1,2      
            )
            SELECT
                User_id_code, Visit_date,
                SUM(visit_tms_daily) OVER
                    (PARTITION BY User_id_code
                    ORDER BY Visit_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                AS visit_tms_cum
            FROM daily
    """)
    con.execute("""
        CREATE OR REPLACE TABLE asof_right AS
            SELECT
                (c1.visit_tms_cum IS NULL) AS no_hist_marker,
                r.receipt_key,
                COALESCE(c1.visit_tms_cum, 0) AS visit_tms_right
            FROM receipts r
            ASOF LEFT JOIN cum c1
                ON r.User_id_code = c1.User_id_code
                AND r.Receive_date - INTERVAL 1 DAY >= c1.Visit_date
    """)
    for window_len in lookback_days:
        con.execute(f"""
            CREATE OR REPLACE TABLE asof_left AS
                SELECT
                    (c2.visit_tms_cum IS NULL) AS no_hist_marker,
                    r.receipt_key,
                    COALESCE(c2.visit_tms_cum, 0) AS visit_tms_left
                FROM receipts r
                ASOF LEFT JOIN cum c2
                    ON r.User_id_code = c2.User_id_code
                    AND r.Receive_date - INTERVAL {window_len} DAY >= c2.Visit_date
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE rlookback AS
                SELECT a.receipt_key,
                    (a.no_hist_marker OR b.no_hist_marker)::INT AS no_hist_marker,
                    (a.visit_tms_right - b.visit_tms_left) AS visit_tms_{window_len-1}d
                FROM asof_right a
                LEFT JOIN asof_left b
                    ON a.receipt_key = b.receipt_key
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE receipts AS
                SELECT r.*,
                    bk.no_hist_marker AS no_hist_visits_marker_{window_len-1}d,
                    ROUND(bk.visit_tms_{window_len-1}d / {window_len-1}, 3)
                        AS Freq_visit_{window_len-1}d
                FROM receipts r
                LEFT JOIN rlookback bk
                    ON r.receipt_key = bk.receipt_key
                ORDER BY
                    r.receipt_key
        """)
    
    # ===================================
    # Section 3: Write parquet & close
    # ===================================
    con.sql("SELECT * FROM receipts").write_parquet(str(out_parquet))
    con.close()