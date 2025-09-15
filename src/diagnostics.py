# src/diagnostics.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd

## notes on the receipt.parquet:
### receipt:
    # added receipt_key, 
    # added labels and audit counts,
    # no dup in rows.

def diag_on_receipts(
    receipts_parquet: Path,
    Price_limit_bin_splits: list,  # in units of cents
    Coupon_limit_bin_splits: list, # in units of cents
    Expiry_span_bin_splits: list,
    threads: int = 8,
) -> pd.DataFrame:
    """
    After segmenting receipts by the price limit, coupon limit and expiry span,
    calculate the percentage of valid coupons and redemption rate (st & fh).
    
    Output: a panda dataframe with each row representing a segment.
        indices: Price_limit_bin, Coupon_limit_bin, and Expiry_span_bin.
        cols: validity percentage, the validity percentage, short-term and full horizon redemption rate.
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # =================
    # Section 1: Load
    # =================
    con.execute("""
        CREATE OR REPLACE TABLE receipts AS
            SELECT 
                Price_limit_cent AS Price_limit,
                Coupon_amt_cent AS Coupon_limit,
                CASE 
                    WHEN Start_date IS NOT NULL AND End_date IS NOT NULL AND End_date >= Start_date
                    THEN End_date - Start_date 
                    ELSE NULL 
                END AS Expiry_span,
                label_valid AS L_valid,
                label_same_user_fh AS L_usage_fh,
                label_same_user_st AS L_usage_st
            FROM read_parquet(?)""", [str(receipts_parquet)])
    
    # =======================================
    # Section 2: Segmentation & Statistics
    # =======================================
    def _set_lower_upper_bounds(i: int, splits: list, 
                                minv: int = 0, maxv: int = 100000000000000) -> tuple[int]:
        if i != 0:
            lower = splits[i - 1]
        else:
            lower = minv
        if i != len(splits):
            upper = splits[i]
        else:
            upper = maxv
        return (lower, upper)
    
    con.execute("""
        CREATE TABLE segs (Price_limit_bin_code INTEGER, Coupon_limit_bin_code INTEGER, Expiry_span_bin_code INTEGER,
                            validity_pc DOUBLE, redeem_fh_pc DOUBLE, redeem_st_pc DOUBLE);
    """)

    for i in range(len(Price_limit_bin_splits) + 1):
        Price_limit_lower, Price_limit_upper = _set_lower_upper_bounds(i, Price_limit_bin_splits)

        for j in range(len(Coupon_limit_bin_splits) + 1):
            Coupon_limit_lower, Coupon_limit_upper = _set_lower_upper_bounds(j, Coupon_limit_bin_splits)
            
            for k in range(len(Expiry_span_bin_splits) + 1):
                Expiry_span_lower, Expiry_span_upper = _set_lower_upper_bounds(k, Expiry_span_bin_splits, maxv=365)
                
                con.execute(f"""
                    CREATE OR REPLACE TABLE seg AS
                        SELECT
                            {i} AS Price_limit_bin_code,
                            {j} AS Coupon_limit_bin_code,
                            {k} AS Expiry_span_bin_code,
                            ROUND(SUM(L_valid) / COUNT(*) * 100, 2) AS validity_pc,
                            ROUND(SUM(L_usage_fh) / COUNT(*) * 100, 2) AS redeem_fh_pc,
                            ROUND(SUM(L_usage_st) / COUNT(*) * 100, 2) AS redeem_st_pc
                        FROM receipts
                        WHERE
                            Price_limit BETWEEN {Price_limit_lower} AND {Price_limit_upper}
                            Coupon_limit BETWEEN {Coupon_limit_lower} AND {Coupon_limit_upper}
                            Expiry_span BETWEEN {Expiry_span_lower} AND {Expiry_span_upper}
                """)

                con.execute("""
                    CREATE OR REPLACE TABLE segs AS
                        SELECT * FROM segs
                        UNION BY NAME
                        SELECT * FROM seg
                """)
    
    # ==================================
    # Section 3: output as pd.DataFrame
    # ==================================
    df = con.sql("SELECT * FROM segs").df()
    return df


# TODO: correct the logic of repeat redemption in labels.py and then calculate the repeat redemption rate per segment.