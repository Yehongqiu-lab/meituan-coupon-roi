# src/io_load.py
import os
import pandas as pd

def get_repo_root():
    curr_dir = os.getcwd()
    repo_root = os.path.dirname(curr_dir)
    return repo_root

def _require(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}")

def load_transactions_csv(path="data_raw/order_detail.csv", nrows=None) -> pd.DataFrame:
    # load data
    name = "txn"
    path = os.path.join(get_repo_root(), path)
    df = pd.read_csv(path, dtype="string", low_memory=False, nrows=nrows)

    # check required columns
    req = ["User_id","Shop_id","Order_id","Coupon_id","Coupon_type",
           "Biz_code","Pay_date","Actual_pay","Reduce_amount"]
    _require(df, req, name)

    # clean and convert types
    df["Pay_date"] = pd.to_datetime(df["Pay_date"], errors="coerce")

    for col in ["Actual_pay","Reduce_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col+"_cent"] = (df[col] * 100).round().astype("Int64")
        df.drop(columns=col, inplace=True)

    c = "Coupon_type"
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df

def load_receipts_csv(path="data_raw/user_coupon_receive.csv", nrows=None) -> pd.DataFrame:
    # load data
    name = "receipt"
    path = os.path.join(get_repo_root(), path)
    df = pd.read_csv(path, dtype="string", low_memory=False, nrows=nrows)

    # check required columns
    req = ["User_id","Coupon_id","Coupon_status","Coupon_amt",
           "Receive_date","Start_date","End_date","Price_limit"]
    _require(df, req, name)

    # clean and convert types
    for c in ["Receive_date","Start_date","End_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in ["Coupon_amt","Price_limit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c+"_cent"] = (df[c] * 100).round().astype("Int64")
        df.drop(columns=c, inplace=True)

    c = "Coupon_status"
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    
    return df

def load_users_logins_csv(path="data_raw/user_visit_detail.csv", nrows=None) -> pd.DataFrame:
    # load data
    name = "user_visit"
    path = os.path.join(get_repo_root(), path)
    df = pd.read_csv(path, dtype="string", low_memory=False, nrows=nrows)

    # check required columns
    req = ["User_id","Visit_date"]
    _require(df, req, name)

    # clean and convert types
    df["Visit_date"] = pd.to_datetime(df["Visit_date"], errors="coerce")
    
    return df

def save_df2pq(df, name):
    repo_root = get_repo_root()
    path = os.path.join(repo_root, f"data_work/{name}.parquet")
    df.to_parquet(path, 
                engine="pyarrow",
                compression="snappy",  # 'zstd' more space-efficient but slower
                index=False)
    print(f"[{name}] saved to {path}, shape={df.shape}, size={os.path.getsize(path)/1024**2:.2f} MB")

def load_df_from_pq(path, cols="all", **kargs) -> pd.DataFrame:
    repo_root = get_repo_root()
    path = os.path.join(repo_root, path)
    if cols != "all":
        df = pd.read_parquet(path, engine="pyarrow", columns=cols, **kargs)
    else:
        df = pd.read_parquet(path, engine="pyarrow", **kargs)
    print(f"pq loaded from {path}, shape={df.shape}, size={os.path.getsize(path)/1024**2:.2f} MB")
    return df