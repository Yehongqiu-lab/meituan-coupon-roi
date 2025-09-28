# src/cpn_features.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd
from datetime import date
import datetime
from datetime import date

holidays = [date(2023, 1, 1), date(2023, 1, 2), 
           date(2023, 1, 21), date(2023, 1, 22), date(2023, 1, 23), date(2023, 1, 24), date(2023, 1, 25), date(2023, 1, 26), date(2023, 1, 27),
           date(2023, 4, 5),
           date(2023, 4, 29), date(2023, 4, 30), date(2023, 5, 1), date(2023, 5, 2), date(2023, 5, 3),
           date(2023, 6, 22), date(2023, 6, 23), date(2023, 6, 24)]
workdays = [date(2023, 1, 28), date(2023, 1, 29), date(2023, 4, 23), date(2023, 5, 6), date(2023, 6, 25)]

## coupon features
    # Encode bins on Coupon_amt_cent, Price_limit_cent, (End_date - Start_date); numeric + categorical
    # generosity ratio = Coupon_amt_cent / (Price_limit_cent + 1)
    # the time difference between Receive_date and Start_date: (Start_date - Receive_date)
    # the time difference between End_date and Receive_date: (End_date - Receive_date)
    # the flags of whether the coupon is received on weekday/workday or weekend/holiday
    # the HISTORICAL invalidity rate of the coupon's segment
    # the HISTORICAL redemption rate of the coupon's segment
def coupon_features(
        receipts_labelled_parquet: Path,
        out_parquet: Path,
        lookback_days: list[int] = [7, 14, 30],
        Price_limit_bin_splits: list[int] = [1000, 10000], # only 2 splits allowed; in units of cents
        Coupon_limit_bin_splits: list[int] = [1000, 10000], # only 2 splits allowed; in units of cents
        Expiry_span_bin_split: int = 10, # only 1 split allowed; in units of days
        holidays: list[datetime.date] = holidays,
        workdays: list[datetime.date] = workdays,
        threads: int = 8
) -> None:
    """
    Generate coupon features and save to out_parquet."""
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # =================
    # Section 1: Load
    # =================
    con.execute("""
        CREATE OR REPLACE TABLE receipts AS
            SELECT
                receipt_key,
                User_id_code,
                Coupon_id_code,
                Price_limit_cent,
                Coupon_amt_cent,
                Receive_date, Start_date, End_date,
                CASE WHEN label_valid = 1 THEN 0 ELSE 1 END AS label_invalid,
                label_same_user_fh,
                label_same_user_st
            FROM read_parquet(?)""", [str(receipts_labelled_parquet)])

    # =====================================================
    # Section 2.1: Create bins + categorical features for
    # Price_limit_cent, Coupon_amt_cent, Expiry_span
    # =====================================================
    con.execute(f"""
        CREATE OR REPLACE TABLE receipts_1 AS     
        SELECT *,
            CASE 
                WHEN Price_limit_cent <= {Price_limit_bin_splits[0]} THEN 0
                WHEN Price_limit_cent > {Price_limit_bin_splits[0]} AND Price_limit_cent <= {Price_limit_bin_splits[1]} THEN 1
                ELSE 2 END AS Price_limit_bin,
            CASE
                WHEN Coupon_amt_cent <= {Coupon_limit_bin_splits[0]} THEN 0
                WHEN Coupon_amt_cent > {Coupon_limit_bin_splits[0]} AND Coupon_amt_cent <= {Coupon_limit_bin_splits[1]} THEN 1
                ELSE 2 END AS Coupon_limit_bin,
            CASE
                WHEN End_date BETWEEN Start_date AND (INTERVAL {Expiry_span_bin_split} DAY + Start_date) THEN 0
                WHEN End_date > (INTERVAL {Expiry_span_bin_split} DAY + Start_date) THEN 1
                ELSE -1 END AS Expiry_span_bin
        FROM receipts
    """)
    
    # ===================================================
    # Section 2.2: Create generosity ratio, 
    # various date precedence markers, 
    # and the weekday/workday v.s. weekend/holiday flag
    # ===================================================
    con.execute("""
        CREATE OR REPLACE TABLE receipts_2 AS
            SELECT *,
                Coupon_amt_cent / (Price_limit_cent + 1) AS Generosity_ratio,
                CASE 
                    WHEN Start_date >= Receive_date THEN 0
                    WHEN Start_date < Receive_date THEN 1
                    ELSE NULL END AS Start_bf_receive_marker,
                CASE 
                    WHEN End_date >= Receive_date THEN 0
                    WHEN End_date < Receive_date THEN 1
                    ELSE NULL END AS End_bf_receive_marker
            FROM receipts_1
    """)
    
    con.execute("CREATE TEMP TABLE workdays(d TIMESTAMP)")
    con.execute("INSERT INTO workdays SELECT * FROM UNNEST(?::TIMESTAMP[])", [workdays])

    con.execute("CREATE TEMP TABLE holidays(d TIMESTAMP)")
    con.execute("INSERT INTO holidays SELECT * FROM UNNEST(?::TIMESTAMP[])", [holidays])


    con.execute(f"""
        CREATE OR REPLACE TABLE receipts_3 AS
            SELECT r.*,
                CASE
                    WHEN dayname(r.Receive_date) IN ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] THEN 1
                    ELSE 0 END AS Weekday_marker,
                CASE
                    WHEN w.d IS NOT NULL THEN 1 ELSE 0 END AS Workday_marker,
                CASE
                    WHEN h.d IS NOT NULL THEN 1 ELSE 0 END AS Holiday_marker
            FROM receipts_2 r
            LEFT JOIN workdays w ON r.Receive_date = w.d
            LEFT JOIN holidays h ON r.Receive_date = h.d
    """)

    # ===================================================
    # Section 2.3: 
    # the HISTORICAL invalidity & redemption rate of the coupon's segment
    # ===================================================
    con.execute("""CREATE OR REPLACE TABLE receipts_4 AS
                SELECT * from receipts_3
                ORDER BY Receive_date, receipt_key""")

    con.execute("""
        CREATE OR REPLACE TABLE cum AS
            -- pre-aggregate to daily
            WITH daily AS (
                SELECT
                    Receive_date,
                    Price_limit_bin, Coupon_limit_bin, Expiry_span_bin,
                    SUM(label_invalid)      AS daily_invalid,
                    SUM(label_same_user_fh) AS daily_fh_redeem,
                    SUM(label_same_user_st) AS daily_st_redeem,
                    COUNT(*)                AS daily_vol
                FROM receipts_4
                GROUP BY 1,2,3,4
            )
            SELECT 
                Receive_date, 
                Price_limit_bin, Coupon_limit_bin, Expiry_span_bin, 
                SUM(daily_invalid)      OVER same_seg AS cum_invalid,
                SUM(daily_fh_redeem)    OVER same_seg AS cum_fh_redeem,
                SUM(daily_st_redeem)    OVER same_seg AS cum_st_redeem,
                SUM(daily_vol)          OVER same_seg AS cumall
            FROM daily
            WINDOW same_seg AS (
                PARTITION BY Price_limit_bin, Coupon_limit_bin, Expiry_span_bin
                ORDER BY Receive_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW 
            )
    """)

    con.execute("""
        CREATE OR REPLACE TABLE asof_right AS
            SELECT r.Receive_date, r.receipt_key,
                (c1.cumall IS NULL) AS no_history_indicator,
                COALESCE(c1.cum_invalid, 0) AS invalid_right,
                COALESCE(c1.cum_fh_redeem, 0) AS fh_right,
                COALESCE(c1.cum_st_redeem, 0) AS st_right,
                COALESCE(c1.cumall, 0) AS all_right
            FROM receipts_4 AS r
            ASOF LEFT JOIN cum c1
                ON r.Price_limit_bin = c1.Price_limit_bin
                AND r.Coupon_limit_bin = c1.Coupon_limit_bin
                AND r.Expiry_span_bin = c1.Expiry_span_bin
                AND (r.Receive_date - INTERVAL 1 DAY) >= c1.Receive_date
    """)

    for window_len in lookback_days:
        con.execute(f"""
            CREATE OR REPLACE TABLE rlookback AS
            WITH asof_left AS (
                SELECT r.Receive_date, r.receipt_key,
                    (c2.cumall IS NULL) AS no_history_indicator,
                    COALESCE(c2.cum_invalid, 0) AS invalid_left,
                    COALESCE(c2.cum_fh_redeem, 0) AS fh_left,
                    COALESCE(c2.cum_st_redeem, 0) AS st_left,
                    COALESCE(c2.cumall, 0) AS all_left,
                FROM receipts_4 AS r
                ASOF LEFT JOIN cum c2
                    ON r.Price_limit_bin = c2.Price_limit_bin
                    AND r.Coupon_limit_bin = c2.Coupon_limit_bin
                    AND r.Expiry_span_bin = c2.Expiry_span_bin
                    AND (r.Receive_date - INTERVAL {window_len} DAY) >= c2.Receive_date
            )
        
            SELECT
                a.receipt_key, a.Receive_date,
                (a.no_history_indicator OR b.no_history_indicator)::INT AS no_history_indicator,
                (a.invalid_right - b.invalid_left) AS invalid_{window_len}d,
                (a.fh_right - b.fh_left) AS fh_{window_len}d,
                (a.st_right - b.st_left) AS st_{window_len}d,
                (a.all_right - b.all_left) AS row_count_{window_len}d
            FROM asof_right a
            LEFT JOIN asof_left b
                ON a.receipt_key = b.receipt_key
        """)

        con.execute(f"""
            CREATE OR REPLACE TABLE receipts_4 AS
            SELECT r.*, 
                bk.no_history_indicator AS no_history_indicator_{window_len}d,
                CASE 
                    WHEN bk.row_count_{window_len}d = 0 THEN 0.000 
                    ELSE ROUND(bk.invalid_{window_len}d / bk.row_count_{window_len}d, 3) 
                        END AS Rate_invalid_{window_len}d,
                CASE
                    WHEN bk.row_count_{window_len}d = 0 THEN 0.000
                    ELSE ROUND(bk.fh_{window_len}d / bk.row_count_{window_len}d, 3)
                        END AS Rate_fh_redeem_{window_len}d,
                CASE
                    WHEN bk.row_count_{window_len}d = 0 THEN 0.000
                    ELSE ROUND(bk.st_{window_len}d / bk.row_count_{window_len}d, 3)
                        END AS Rate_st_redeem_{window_len}d
            FROM receipts_4 r
            LEFT JOIN rlookback bk
                ON r.receipt_key = bk.receipt_key 
        """)
    
    # ===================================
    # Section 3: Write parquet and close
    # ===================================
    con.sql("SELECT * FROM receipts_4").write_parquet(str(out_parquet))
    con.close()
    
    


    


