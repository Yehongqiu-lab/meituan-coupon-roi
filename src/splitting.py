# src/splitting.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd
from datetime import date

# perform various splitting strategies for different models:
    # right-censored data awareness
    # return the cv-partitioned train/val sets and the testing set
    # splitting integrity: apply lookback guard/purge period if necessary

# Policy modeling
def policy_model_split(
        trainable_parquet: Path,
        train_set_out_parquet: Path,
        test_set_out_parquet: Path,
        test_start_at: date = date(2023, 6, 1),
        right_censoring: bool = True,
        censor_cutoff_at: date = date(2023, 6, 30),
        forward_chain_cv: bool = True,
        threads: int = 8
):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # =====================
    # Section 1: Load
    # =====================
    con.execute("""
        CREATE OR REPLACE TABLE trainable AS
            SELECT * FROM read_parquet(?)
    """, [str(trainable_parquet)])

    # ======================================================
    # Section 2: Applying right censoring if required
    # ======================================================
    if right_censoring:
        con.execute("""
            CREATE OR REPLACE TABLE right_censored_trainable AS
                SELECT * FROM trainable
                WHERE
                    End_date <= ?
        """, [censor_cutoff_at])

    # =======================================================
    # Section 3: Splitting the train set and the test set
    # =======================================================
    con.execute("""
        CREATE OR REPLACE TABLE test AS
            SELECT * FROM right_censored_trainable
            WHERE
                Receive_date >= ?
    """, [test_start_at])
    # data integrity: apply lookback guard on the train set:
    con.execute("""
        CREATE OR REPLACE TABLE train AS
            SELECT * FROM right_censored_trainable
            WHERE
                End_date < ?
    """, [test_start_at])

    con.sql("SELECT * FROM test").write_parquet(str(test_set_out_parquet))

    # ============================================================
    # Section 4: Partition 3 cv folds on the train set if required
    # ============================================================
    if forward_chain_cv:
        con.execute("""
            CREATE OR REPLACE TABLE folds AS
                SELECT *, 
                    CASE 
                        WHEN Receive_date BETWEEN '2023-03-16' AND '2023-03-31' THEN 2
                        WHEN End_date < '2023-03-16' THEN 1
                        ELSE 0 END AS fold_1_marker,
                    CASE
                        WHEN Receive_date BETWEEN '2023-04-16' AND '2023-04-30' THEN 2
                        WHEN End_date < '2023-04-16' THEN 1
                        ELSE 0 END AS fold_2_marker,
                    CASE
                        WHEN Receive_date BETWEEN '2023-05-16' AND '2023-05-31' THEN 2
                        WHEN End_date < '2023-05-16' THEN 1
                        ELSE 0 END AS fold_3_marker
                FROM train
        """)
        con.sql("SELECT * FROM folds").write_parquet(str(train_set_out_parquet))
    else:
        con.sql("SELECT * FROM train").write_parquet(str(train_set_out_parquet))
    
    
# ROI modeling
## only support the 15-day short term case as-of now
def ROI_model_split(
        trainable_parquet: Path,
        train_set_out_parquet: Path,
        test_set_out_parquet: Path,
        test_start_at: date = date(2023, 6, 1),
        right_censoring: bool = True,
        censor_cutoff_at: date = date(2023, 6, 30),
        forward_chain_cv: bool = True,
        threads: int = 8
):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    # =====================
    # Section 1: Load
    # =====================
    con.execute("""
        CREATE OR REPLACE TABLE trainable AS
            SELECT * FROM read_parquet(?)
    """, [str(trainable_parquet)])

    # ======================================================
    # Section 2: Applying right censoring if required
    # ======================================================
    if right_censoring:
        con.execute("""
            CREATE OR REPLACE TABLE right_censored_trainable AS
                SELECT * FROM trainable
                WHERE
                    End_date <= ?
        """, [censor_cutoff_at])

    # =======================================================
    # Section 3: Splitting the train set and the test set
    # =======================================================
    con.execute("""
        CREATE OR REPLACE TABLE test AS
            SELECT * FROM right_censored_trainable
            WHERE
                Receive_date >= ?
    """, [test_start_at])
    # data integrity: apply a 15-day purge on the train set:
    con.execute("""
        CREATE OR REPLACE TABLE train AS
            SELECT * FROM right_censored_trainable
            WHERE
                Receive_date < ? - INTERVAL 15 DAY
    """, [test_start_at])

    con.sql("SELECT * FROM test").write_parquet(str(test_set_out_parquet))

    # ============================================================
    # Section 4: Partition 3 cv folds on the train set if required
    # ============================================================
    if forward_chain_cv:
        con.execute("""
            CREATE OR REPLACE TABLE folds AS
                SELECT *, 
                    CASE 
                        WHEN Receive_date BETWEEN '2023-03-16' AND '2023-03-31' THEN 2
                        WHEN Receive_date < (TIMESTAMP '2023-03-16' - INTERVAL 15 DAY) THEN 1
                        ELSE 0 END AS fold_1_marker,
                    CASE
                        WHEN Receive_date BETWEEN '2023-04-16' AND '2023-04-30' THEN 2
                        WHEN Receive_date < (TIMESTAMP '2023-04-16' - INTERVAL 15 DAY) THEN 1
                        ELSE 0 END AS fold_2_marker,
                    CASE
                        WHEN Receive_date BETWEEN '2023-05-16' AND '2023-05-31' THEN 2
                        WHEN Receive_date < (TIMESTAMP '2023-05-16' - INTERVAL 15 DAY) THEN 1
                        ELSE 0 END AS fold_3_marker
                FROM train
        """)
        con.sql("SELECT * FROM folds").write_parquet(str(train_set_out_parquet))
    else:
        con.sql("SELECT * FROM train").write_parquet(str(train_set_out_parquet))
    