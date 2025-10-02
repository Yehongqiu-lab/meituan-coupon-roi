import pyarrow.parquet as pq
import pandas as pd
from src.user_features import user_features

def test_one_user_longer_time_than_window(make_labelled_receipts, add_receipt_keys,
                            make_txns, make_visits,
                            cast_datatype, to_parquet):
    """
    Case 1: When there is only one user visiting, receiving coupons, and make purchases,
    all over a period of >2d (lookback window). 
    Test if all user features work correctly,
    especially the several no_hist_markers' behavior."""
    # construct data
    receipt_rows = (
       (1, 9001, 100, 0, "2023-01-01", "2023-01-02", "2023-01-03", 1, 0, 0),
       (1, 9002, 900, 0, "2023-01-01", "2023-01-02", "2023-01-03", 1, 1, 1),
       (1, 9003, 500, 1000, "2023-01-02", "2023-01-02", "2023-01-03", 0, 1, 1),
       (1, 9004, 250, 1000, "2023-01-03", "2023-01-03", "2023-01-03", 1, 0, 0),
       (1, 9005, 100, 0, "2023-01-03", "2023-01-04", "2023-01-05", 1, 1, 1))
    rcs = make_labelled_receipts(*receipt_rows)
    rcs = add_receipt_keys(rcs, keys=range(1, 6))
    rcs = cast_datatype(rcs, "receipt_labelled")
    rp = "tests/data_test/rcs_labelled_1.parquet"; to_parquet(rcs, rp)

    txn_rows = (
	    (1, -1, "2023-01-01", 2500, 0, 1),
	    (1, 9002, "2023-01-01", 2500, 900, 1),
	    (1, -1, "2023-01-02", 2500, 0, 2),
	    (1, -1, "2023-01-03", 1200, 0, 3))
    txns = make_txns(*txn_rows)
    txns = cast_datatype(txns, "txn_wo_key")
    tp = "tests/data_test/txns_1.parquet"; to_parquet(txns, tp)

    visit_rows = (              
	    (1, "2023-01-01"),
	    (1, "2023-01-02"), 
	    (1, "2023-01-03"))
    visits = make_visits(*visit_rows)
    visits = cast_datatype(visits, "visit")
    vp = "tests/data_test/visits_1.parquet"; to_parquet(visits, vp)

    # go through the function:
    outp = "tests/data_test/rcs_labelled_featureout_1.parquet"
    user_features(rp, tp, vp, outp, lookback_days=[2])

    # assertion:
    out = pq.read_table(outp).to_pandas()

    ## length:
    assert len(out) == 5

    ## no history markers
    no_hist_marker_ref = [1, 1, 1, 0, 0]
    colnames = ["no_hist_rcs_marker_2d", "no_hist_txns_marker_2d", "no_hist_visits_marker_2d"]
    for col in colnames:
        assert (out[col] == no_hist_marker_ref).all()

    ## historical invalidity/redemption rate of the same user:
    invalid_ref = [0, 0, 0, 1, 1]
    fh_ref = [0, 0, 1/2, 1, 1] # st_ref = fh_ref
    
    for i, value in enumerate(out["Rate_same_user_invalid_2d"]):
        assert abs(value - invalid_ref[i]) < 1e-3
    for i, value in enumerate(out["Rate_same_user_fh_redeem_2d"]):
        assert abs(value - fh_ref[i]) < 1e-3
    assert (out["Rate_same_user_fh_redeem_2d"] == out["Rate_same_user_st_redeem_2d"]).all()

    ## historical avgspend vs pricelimit, avg reduce amt vs coupon amt, of the same user:
    ref1 = [0, 0, 2500/1001, 2500/1001, 2500]
    ref2 = [0, 0, 900/501, 0, 0]
    for i, value in enumerate(out["Rt_avgspend_vs_pricelimit_2d"]):
        assert abs(value - ref1[i]) < 1e-3
    for i, value in enumerate(out["Rt_avgreduce_vs_couponamt_2d"]):
        assert abs(value - ref2[i]) < 1e-3
    
    ## historical freq of purchase & visit of the same user:
    fr = [0, 0, 1, 1, 1]
    for i, value in enumerate(out["Freq_purchase_2d"]):
        assert abs(value - fr[i]) < 1e-3
    assert (out["Freq_visit_2d"] == out["Freq_purchase_2d"]).all()

    

    
    




def test_multi_user_short_time():
    ...