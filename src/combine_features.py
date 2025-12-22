# src/combine_features.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd
from datetime import date
import datetime

def features_combining(
        cpn_features_parquet: Path,
        user_features_parquet: Path,
        out_parquet: Path,
        threads: int = 8
) -> None:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    con.execute("""
        CREATE OR REPLACE TABLE rcs_cpn_fea AS
            SELECT
                receipt_key,
                Receive_date, Start_date, End_date,
                Price_limit_cent, Coupon_amt_cent, 
                Price_limit_bin, Coupon_limit_bin, Expiry_span_bin,
                Generosity_ratio, 
                Start_bf_receive_marker, End_bf_receive_marker,
                Weekday_marker, Workday_marker, Holiday_marker,
                no_history_indicator_7d, Rate_invalid_7d, Rate_fh_redeem_7d, Rate_st_redeem_7d,
                no_history_indicator_14d, Rate_invalid_14d, Rate_fh_redeem_14d, Rate_st_redeem_14d,
                no_history_indicator_30d, Rate_invalid_30d, Rate_fh_redeem_30d, Rate_st_redeem_30d
            FROM read_parquet(?)""", [str(cpn_features_parquet)])
    
    con.execute("""
        CREATE OR REPLACE TABLE labels AS
            SELECT
                receipt_key,
                label_invalid, label_same_user_fh, label_same_user_st
            FROM read_parquet(?)""", [str(cpn_features_parquet)])
    
    con.execute("""
        CREATE OR REPLACE TABLE rcs_user_fea AS
            SELECT
                receipt_key,
                no_hist_rcs_marker_7d, Rate_same_user_invalid_7d, Rate_same_user_fh_redeem_7d, Rate_same_user_st_redeem_7d,
                no_hist_rcs_marker_14d, Rate_same_user_invalid_14d, Rate_same_user_fh_redeem_14d, Rate_same_user_st_redeem_14d,
                no_hist_rcs_marker_30d, Rate_same_user_invalid_30d, Rate_same_user_fh_redeem_30d, Rate_same_user_st_redeem_30d, 
                no_hist_txns_marker_7d, Rt_avgspend_vs_pricelimit_7d, Rt_avgreduce_vs_couponamt_7d, Freq_purchase_7d,
                no_hist_txns_marker_14d, Rt_avgspend_vs_pricelimit_14d, Rt_avgreduce_vs_couponamt_14d, Freq_purchase_14d,
                no_hist_txns_marker_30d, Rt_avgspend_vs_pricelimit_30d, Rt_avgreduce_vs_couponamt_30d, Freq_purchase_30d,
                no_hist_visits_marker_7d, Freq_visit_7d,
                no_hist_visits_marker_14d, Freq_visit_14d,
                no_hist_visits_marker_30d, Freq_visit_30d
            FROM read_parquet(?)""", [str(user_features_parquet)])
    
    con.execute("""
        CREATE OR REPLACE TABLE trainable AS
            SELECT * FROM rcs_cpn_fea c
            LEFT JOIN rcs_user_fea u ON c.receipt_key = u.receipt_key
            LEFT JOIN labels l ON c.receipt_key = l.receipt_key
    """)

    con.sql("SELECT * FROM trainable").write_parquet(str(out_parquet))
    con.close()