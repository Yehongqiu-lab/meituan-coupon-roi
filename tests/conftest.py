# tests/conftest.py
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

@pytest.fixture
def make_txn():
    def _make_txn(user, coupon, pay_date, pay_amt_cent, reduce_amt_cent, 
                  order=1, shop=1, biz_code='A', coupon_type=1):
        return pd.DataFrame([{
            "User_id_code": user,
            "Shop_id_code": shop,
            "Order_id_code": order,
            "Coupon_id_code": coupon,
            "Coupon_type": coupon_type,
            "Biz_code": biz_code,
            "Pay_date": pd.Timestamp(pay_date),
            "Actual_pay_cent": pay_amt_cent,
            "Reduce_amount_cent": reduce_amt_cent
        }])
    return _make_txn

@pytest.fixture
def make_txns(make_txn):
    def _make_txns(*rows):
        return pd.concat([make_txn(*row) for row in rows], ignore_index=True)
    return _make_txns

@pytest.fixture
def make_receipt():
    def _make_receipt(user, coupon, coupon_amt_cent, receive_date, start_date, end_date, 
                     price_limit_cent=1000, coupon_status=1):
        return pd.DataFrame([{
            "User_id_code": user,
            "Coupon_id_code": coupon,
            "Coupon_amt_cent": coupon_amt_cent,
            "Receive_date": pd.Timestamp(receive_date),
            "Start_date": pd.Timestamp(start_date),
            "End_date": pd.Timestamp(end_date),
            "Price_limit_cent": price_limit_cent,
            "Coupon_status": coupon_status
        }])
    return _make_receipt

@pytest.fixture
def make_receipts(make_receipt):
    def _make_receipts(*rows):
        return pd.concat([make_receipt(*row) for row in rows], ignore_index=True)
    return _make_receipts

@pytest.fixture
def make_labelled_receipt():
    def _make_labelled_receipt(user, coupon, coupon_amt_cent, price_limit_cent, 
                               receive_date, start_date, end_date, 
                               label_valid, label_fh_redeem, label_st_redeem):
        return pd.DataFrame([{
            "User_id_code": user,
            "Coupon_id_code": coupon,
            "Coupon_amt_cent": coupon_amt_cent,
            "Price_limit_cent": price_limit_cent,
            "Receive_date": pd.Timestamp(receive_date),
            "Start_date": pd.Timestamp(start_date),
            "End_date": pd.Timestamp(end_date),
            "label_valid": label_valid,
            "label_same_user_fh": label_fh_redeem,
            "label_same_user_st": label_st_redeem
        }])
    return _make_labelled_receipt

@pytest.fixture
def make_labelled_receipts(make_labelled_receipt):
    def _make_labelled_receipts(*rows):
        return pd.concat([make_labelled_receipt(*row) for row in rows], ignore_index=True)
    return _make_labelled_receipts

@pytest.fixture
def make_visit():
    def _make_visit(user, visit_date):
        return pd.DataFrame([{
            "User_id_code": user,
            "Visit_date": pd.Timestamp(visit_date)
        }])
    return _make_visit

@pytest.fixture
def make_visits(make_visit):
    def _make_visits(*rows):
        return pd.concat([make_visit(*row) for row in rows], ignore_index=True)
    return _make_visits

@pytest.fixture
def add_txn_key():
    def _add_txn_key(txn_df, key=1):
        txn_df = txn_df.assign(txn_key = key)
        return txn_df
    return _add_txn_key

@pytest.fixture
def add_txn_keys():
    def _add_txn_keys(txns_df, keys):
        txns_df = txns_df.assign(txn_key = keys)
        return txns_df
    return _add_txn_keys

@pytest.fixture
def add_receipt_key():
    def _add_receipt_key(receipt_df, key=1):
        receipt_df = receipt_df.assign(receipt_key = key)
        return receipt_df
    return _add_receipt_key

@pytest.fixture
def add_receipt_keys():
    def _add_receipt_keys(receipts_df, keys):
        receipts_df = receipts_df.assign(receipt_key = keys)
        return receipts_df
    return _add_receipt_keys

@pytest.fixture
def cast_datatype():
    """Note: cast_datatype is used until key(s) is/are added."""
    def _cast_datetype(df, flag):
        if flag == "txn":
            to_Int_cols = ["txn_key", "User_id_code", "Shop_id_code", 
                              "Coupon_id_code",
                              "Order_id_code", "Coupon_type",
                              "Actual_pay_cent", "Reduce_amount_cent"]
        elif flag == "txn_wo_key":
            to_Int_cols = ["User_id_code", "Shop_id_code", 
                              "Coupon_id_code",
                              "Order_id_code", "Coupon_type",
                              "Actual_pay_cent", "Reduce_amount_cent"]
        elif flag == "receipt":
            to_Int_cols = ["receipt_key", "User_id_code", "Coupon_id_code",
                           "Coupon_amt_cent", "Price_limit_cent",
                           "Coupon_status"]
        elif flag == "receipt_labelled":
            to_Int_cols = ["receipt_key", "User_id_code", "Coupon_id_code",
                           "Coupon_amt_cent", "Price_limit_cent",
                           "label_valid", "label_same_user_fh", "label_same_user_st"]
        elif flag == "visit":
            to_Int_cols = ["User_id_code"]
            
        for col in to_Int_cols:
            df[col] = df[col].astype("Int64")
        return df
    return _cast_datetype

@pytest.fixture
def to_parquet():
    def _to_parquet(df, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)
    return _to_parquet

