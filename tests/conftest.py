# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def make_txn():
    def _make_txn(user, coupon, pay_date, pay_amt_cent, reduce_amt_cent, 
                  shop=1, order=1, biz_code='A', coupon_type=1):
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
def make_txns():
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
def make_receipts():
    def _make_receipts(*rows):
        return pd.concat([make_receipt(*row) for row in rows], ignore_index=True)
    return _make_receipts

