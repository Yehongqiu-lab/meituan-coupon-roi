import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from src.cpn_features import coupon_features

def test_one_segment(make_labelled_receipts, add_receipt_keys,
                     cast_datatype, to_parquet):
    """
    Case 1: When there are multiple coupon receipts recorded within each day, and
    when all receipts belong to the same coupon segment, 
    test whether the historical rates within
    the lookback window are calculated correctly.
    """
    # construct data
    receipt_rows = (
        (1, 9001, 100, 0, "2023-01-01", "2023-01-02", "2023-01-03", 1, 0, 0),
        (1, 9002, 900, 0, "2023-01-01", "2023-01-02", "2023-01-03", 1, 1, 1),
        (2, 9003, 500, 1000, "2023-01-02", "2023-01-02", "2023-01-09", 0, 1, 1),
        (3, 9004, 250, 1000, "2023-01-03", "2023-01-03", "2023-01-03", 1, 0, 0),
        (4, 9005, 100, 0, "2023-01-03", "2023-01-04", "2023-01-05", 1, 1, 1),
        (5, 9006, 500, 1000, "2023-01-04", "2023-01-04", "2023-01-06", 1, 0, 0),
        (5, 9006, 500, 1000, "2023-01-04", "2023-01-04", "2023-01-06", 1, 0, 0),
        (6, 9007, 100, 900, "2023-01-05", "2023-01-06", "2023-01-16", 1, 1, 1),
        (7, 9007, 100, 900, "2023-01-06", "2023-01-06", "2023-01-16", 1, 0, 0),
        (8, 9008, 250, 1000, "2023-01-07", "2023-01-08", "2023-01-08", 1, 0, 0),
        (9, 9010, 700, 1000, "2023-01-08", "2023-01-09", "2023-01-15", 0, 0, 0))
    rcs = make_labelled_receipts(*receipt_rows)
    rcs = add_receipt_keys(rcs, keys=range(1, 12))
    rcs = cast_datatype(rcs, "receipt_labelled")
    rp = "tests/data_test/rcs_labelled_1.parquet"; to_parquet(rcs, rp)

    # go through the function:
    outp = "tests/data_test/rcs_labelled_featureout_1.parquet"
    coupon_features(rp, outp, lookback_days=[7])

    # assertion:
    out = pq.read_table(outp).to_pandas()
    assert len(out) == 11
    assert (out["no_history_indicator_7d"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).all()
    
    ref1 = [0, 0, 0, 1/3, 1/3, 1/5, 1/5, 1/7, 1/8, 1/9, 1/8]
    for i, value in enumerate(out["Rate_invalid_7d"]):
        assert abs(value - ref1[i]) < 1e-3
    
    ref2 = [0, 0, 1/2, 2/3, 2/3, 3/5, 3/5, 3/7, 1/2, 4/9, 3/8]
    for i, value in enumerate(out["Rate_fh_redeem_7d"]):
        assert abs(value - ref2[i]) < 1e-3
    
    assert (out["Rate_fh_redeem_7d"] == out["Rate_st_redeem_7d"]).all()

    ref3 = [100, 900, 500/1001, 250/1001, 100, 500/1001, 
            500/1001, 100/901, 100/901, 250/1001, 700/1001]
    for i, value in enumerate(out["Generosity_ratio"]):
        assert abs(value - ref3[i]) < 1e-3

    assert (out["Start_bf_receive_marker"] == [0] * 11).all()
    assert (out["End_bf_receive_marker"] == [0] * 11).all()
    assert (out["Weekday_marker"] == [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]).all()
    assert (out["Workday_marker"] == [0] * 11).all()
    assert (out["Holiday_marker"] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all()

def test_multi_segment(make_labelled_receipts, add_receipt_keys,
                     cast_datatype, to_parquet):
    """
    Case 2: When there are multiple coupon receipts recorded within each day, and
    when receipts belong to multiple coupon segments, 
    test whether the historical rates within
    the lookback window for each seg are calculated correctly.
    """
    # construct data
    receipt_rows = (
        (1, 9001, 100, 0, "2023-01-01", "2023-01-02", "2023-01-03", 1, 0, 0),
        (1, 9002, 900, 0, "2023-01-01", "2023-01-02", "2023-01-03", 1, 1, 1),
        (2, 9003, 500, 1000, "2023-01-02", "2023-01-02", "2023-01-09", 0, 1, 1),
        (3, 9004, 250, 1000, "2023-01-03", "2023-01-03", "2023-01-03", 1, 0, 0),
        (4, 9005, 100, 0, "2023-01-03", "2023-01-04", "2023-01-05", 1, 1, 1),
        (4, 9011, 1000, 10000, "2023-01-03", "2023-01-04", "2023-01-24", 1, 1, 0),
        (5, 9006, 500, 1000, "2023-01-04", "2023-01-04", "2023-01-06", 1, 0, 0),
        (5, 9006, 500, 1000, "2023-01-04", "2023-01-04", "2023-01-06", 1, 0, 0),
        (6, 9007, 100, 900, "2023-01-05", "2023-01-06", "2023-01-16", 1, 1, 1),
        (7, 9007, 100, 900, "2023-01-06", "2023-01-06", "2023-01-16", 1, 0, 0),
        (8, 9008, 250, 1000, "2023-01-07", "2023-01-08", "2023-01-08", 1, 0, 0),
        (9, 9010, 700, 1000, "2023-01-08", "2023-01-09", "2023-01-15", 0, 0, 0))
    rcs = make_labelled_receipts(*receipt_rows)
    rcs = add_receipt_keys(rcs, keys=range(1, 13))
    rcs = cast_datatype(rcs, "receipt_labelled")
    rp = "tests/data_test/rcs_labelled_2.parquet"; to_parquet(rcs, rp)

    # go through the function:
    outp = "tests/data_test/rcs_labelled_featureout_2.parquet"
    coupon_features(rp, outp, lookback_days=[7])

    # assertion:
    out = pq.read_table(outp).to_pandas()
    assert len(out) == 12
    assert (out["no_history_indicator_7d"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).all()
    
    ref1 = [0, 0, 0, 1/3, 1/3, 0, 1/5, 1/5, 1/7, 1/8, 1/9, 1/8]
    for i, value in enumerate(out["Rate_invalid_7d"]):
        assert abs(value - ref1[i]) < 1e-3
    
    ref2 = [0, 0, 1/2, 2/3, 2/3, 0, 3/5, 3/5, 3/7, 1/2, 4/9, 3/8]
    for i, value in enumerate(out["Rate_fh_redeem_7d"]):
        assert abs(value - ref2[i]) < 1e-3
    
    assert (out["Rate_fh_redeem_7d"] == out["Rate_st_redeem_7d"]).all()

    assert (out["Coupon_limit_bin"] == [0] * 12).all()
    
    for i, value in enumerate(out["Price_limit_bin"]):
        if i == 5:
            assert value == 1
        else:
            assert value == 0
    
    for i, value in enumerate(out["Expiry_span_bin"]):
        if i == 5:
            assert value == 1
        else:
            assert value == 0