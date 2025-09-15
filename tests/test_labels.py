import pandas as pd
import pyarrow.parquet as pq
from src.labels import build_labels

def test_happy_path_full_and_short_labels(make_txn, make_receipt,
                                          add_txn_key, add_receipt_key,
                                          cast_datatype, to_parquet):
    """
    Case 1.1: Happy path - a received coupon is used in a valid way by the same user;
    the coupon valid window shorter than 15d.
    Expect:
    label_same_user_fh = 1
    label_same_user_st = 1
    label_valid = 1
    """
    # Given: one txn with Coupon_id_code present, positive pay, positive reduce
    txn = make_txn(
        user=1, 
        coupon=9001, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    txn = add_txn_key(txn, key=1)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn.parquet"
    to_parquet(txn, tp)

    # And: one matching receipt for the same user with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=9001,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-09",
        end_date="2023-01-15",
        price_limit_cent=1000,
        coupon_status=1
    )
    receipt = add_receipt_key(receipt, key=11)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt.parquet"
    to_parquet(receipt, rp)

    # When:
    outp = "tests/data_test/labels_out.parquet"
    build_labels(
        receipts_parquet=rp,
        txns_parquet=tp,
        out_parquet=outp,
        short_days=15,
        threads=1
    )
    out = pq.read_table(outp).to_pandas()

    # Then (asserts): 

    # basic info:
    assert len(out) == 1
    row = out.iloc[0]
    assert row["receipt_key"] == 11
    assert row["User_id_code"] == 1
    assert row["Coupon_id_code"] == 9001

    # Dates and effective windows:
    assert pd.Timestamp(row["Receive_date"]) == pd.Timestamp("2023-01-05")
    assert pd.Timestamp(row["Start_date"]) == pd.Timestamp("2023-01-09")
    assert pd.Timestamp(row["End_date"]) == pd.Timestamp("2023-01-15")
    assert pd.Timestamp(row["start_eff"]) == pd.Timestamp("2023-01-09")
    assert pd.Timestamp(row["end_eff"]) == pd.Timestamp("2023-01-15")
    assert pd.Timestamp(row["short_end"]) == pd.Timestamp("2023-01-15")

    # Labels:
    assert row["label_same_user_fh"] == 1
    assert row["label_same_user_st"] == 1
    assert row["label_valid"] == 1

    # Additional fields passthrough:
    assert row["Coupon_status"] == 1
    assert row["Coupon_amt_cent"] == 500
    assert row["Price_limit_cent"] == 1000

    # First valid txn:
    assert row["first_valid_txn_key"] == 1
    assert pd.Timestamp(row["first_valid_txn_time"]) == pd.Timestamp("2023-01-10")

    # Audit counts:
    assert row["same_user_valid_txn_count"] == 1
    assert row["same_user_early_txn_count"] == 0
    assert row["same_user_late_txn_count"] == 0
    assert row["other_user_in_window_txn_count"] == 0
    assert row["other_user_without_own_receipt_txn_count"] == 0

    # Usage validity flags:
    assert row["flag_early"] == 0
    assert row["flag_late"] == 0
    assert row["flag_cross_user"] == 0
    assert row["flag_struc_invalid"] == 0

# ----------------------------
# Case 1.2
# Happy path: valid redemption after 15d; long window (>15d) → fh=1, st=0
# ----------------------------
def test_happy_path_long_window_short_term_zero(make_txn, make_receipt,
                                                add_txn_key, add_receipt_key,
                                                cast_datatype, to_parquet):
    txn = make_txn(user=1, coupon=9001, pay_date="2023-01-18",
                   pay_amt_cent=5000, reduce_amt_cent=500)
    txn = add_txn_key(txn, key=2)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_1_2.parquet"; to_parquet(txn, tp)

    # start_eff = max(Receive(01-01), Start(01-01)) = 01-01; end=02-28; short_end = min(end, Receive+15d)=01-16
    receipt = make_receipt(user=1, coupon=9001, coupon_amt_cent=500,
                           receive_date="2023-01-01", start_date="2023-01-01", end_date="2023-02-28",
                           price_limit_cent=1000, coupon_status=1)
    receipt = add_receipt_key(receipt, key=12)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_1_2.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_1_2.parquet"
    build_labels(rp, tp, outp, short_days=15, threads=1)
    out = pq.read_table(outp).to_pandas()
    row = out.iloc[0]

    assert row["label_same_user_fh"] == 1
    assert row["label_same_user_st"] == 0
    assert row["label_valid"] == 1
    assert row["first_valid_txn_key"] == 2

# ----------------------------
# Case 2
# Inclusivity: Pay_date == start_eff == end_eff → fh=1, st=1
# ----------------------------
def test_inclusivity_edges(make_txn, make_receipt,
                           add_txn_key, add_receipt_key,
                           cast_datatype, to_parquet):
    txn = make_txn(user=7, coupon=9007, pay_date="2023-03-10",
                   pay_amt_cent=3000, reduce_amt_cent=300)
    txn = add_txn_key(txn, key=70)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_2.parquet"; to_parquet(txn, tp)

    # start=end=03-10; receive earlier → start_eff=03-10; short_end=min(end, receive+15d)=end
    receipt = make_receipt(user=7, coupon=9007, coupon_amt_cent=300,
                           receive_date="2023-03-01", start_date="2023-03-10", end_date="2023-03-10",
                           price_limit_cent=1000, coupon_status=2)
    receipt = add_receipt_key(receipt, key=71)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_2.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_2.parquet"
    build_labels(rp, tp, outp)
    row = pq.read_table(outp).to_pandas().iloc[0]
    assert row["label_same_user_fh"] == 1
    assert row["label_same_user_st"] == 1
    assert row["label_valid"] == 1
    assert row["first_valid_txn_key"] == 70

# ----------------------------
# Case 3
# Early/Late counts with no in-window redemption
# ----------------------------
def test_early_late_counts(make_txns, make_receipt,
                           add_txn_keys, add_receipt_key,
                           cast_datatype, to_parquet):
    # two txns: one early, one late
    txn = make_txns(
        (2, 9002, "2023-01-04", 4000, 400),
        (2, 9002, "2023-01-21", 4200, 400)
    )
    txn = add_txn_keys(txn, keys=[21, 22])
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_3.parquet"; to_parquet(txn, tp)

    # start_eff=01-10, end=01-20; neither 01-04 nor 01-21 are in-window
    receipt = make_receipt(user=2, coupon=9002, coupon_amt_cent=400,
                           receive_date="2023-01-05", start_date="2023-01-10", end_date="2023-01-20",
                           price_limit_cent=1000, coupon_status=1)
    receipt = add_receipt_key(receipt, key=23)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_3.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_3.parquet"
    build_labels(rp, tp, outp)
    row = pq.read_table(outp).to_pandas().iloc[0]
    assert row["label_same_user_fh"] == 0
    assert row["label_same_user_st"] == 0
    assert row["label_valid"] == 0

    assert row["flag_early"] == 1
    assert row["flag_late"] == 1

    assert row["same_user_early_txn_count"] == 1
    assert row["same_user_late_txn_count"] == 1
    assert row["same_user_valid_txn_count"] == 0

# ----------------------------
# Case 4
# NULL date fields → labels 0, counts 0 (no crash)
# ----------------------------
def test_null_dates_label_zero(make_txn, make_receipt,
                               add_txn_key, add_receipt_key,
                               cast_datatype, to_parquet):
    txn = make_txn(user=3, coupon=9003, pay_date="2023-01-10",
                   pay_amt_cent=3000, reduce_amt_cent=300)
    txn = add_txn_key(txn, key=31)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_4.parquet"; to_parquet(txn, tp)

    # Start_date is NULL
    receipt = make_receipt(user=3, coupon=9003, coupon_amt_cent=300,
                           receive_date="2023-01-01", start_date=None, end_date="2023-01-31",
                           price_limit_cent=1000, coupon_status=1)
    receipt = add_receipt_key(receipt, key=32)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_4.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_4.parquet"
    build_labels(rp, tp, outp)
    row = pq.read_table(outp).to_pandas().iloc[0]
    assert row["label_same_user_fh"] == 0
    assert row["label_same_user_st"] == 0
    assert row["label_valid"] == 0
    assert row["flag_struc_invalid"] == 1
    assert row["same_user_valid_txn_count"] == 0
    assert pd.isna(row["first_valid_txn_key"])
    assert pd.isna(row["first_valid_txn_time"])

# ----------------------------
# Case 5
# Multiple same-user valid txns; earliest first_valid; count tallies all
# ----------------------------
def test_multiple_same_user_valid_txns(make_txns, make_receipt,
                                       add_txn_keys, add_receipt_key,
                                       cast_datatype, to_parquet):
    txn = make_txns(
        (4, 9004, "2023-02-02", 2000, 200),
        (4, 9004, "2023-02-05", 2200, 200),
        (4, 9004, "2023-02-07", 2300, 200)
    )
    txn = add_txn_keys(txn, keys=[41, 42, 43])
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_5.parquet"; to_parquet(txn, tp)

    receipt = make_receipt(user=4, coupon=9004, coupon_amt_cent=200,
                           receive_date="2023-02-01", start_date="2023-02-01", end_date="2023-02-28",
                           price_limit_cent=1000, coupon_status=1)
    receipt = add_receipt_key(receipt, key=44)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_5.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_5.parquet"
    build_labels(rp, tp, outp)
    row = pq.read_table(outp).to_pandas().iloc[0]
    assert row["label_same_user_fh"] == 1
    assert row["label_same_user_st"] == 1
    assert row["label_valid"] == 1
    assert row["same_user_valid_txn_count"] == 3
    assert row["first_valid_txn_key"] == 41
    assert pd.Timestamp(row["first_valid_txn_time"]) == pd.Timestamp("2023-02-02")

# ----------------------------
# Case 6.1
# Two receipts (same user,coupon) but only one txn; that txn is first_valid for both
# ----------------------------
def test_two_receipts_one_txn_same_first_valid(make_txn, make_receipts,
                                               add_txn_key, add_receipt_keys,
                                               cast_datatype, to_parquet):
    txn = make_txn(user=5, coupon=9005, pay_date="2023-03-15",
                   pay_amt_cent=2400, reduce_amt_cent=240)
    txn = add_txn_key(txn, key=51)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_6_1.parquet"; to_parquet(txn, tp)

    # Two receipts whose windows both include 03-15
    receipt = make_receipts(
        (5, 9005, 240, "2023-03-01", "2023-03-10", "2023-03-20"),
        (5, 9005, 240, "2023-03-05", "2023-03-10", "2023-03-20")
    )
    receipt = add_receipt_keys(receipt, keys=[52, 53])
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_6_1.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_6_1.parquet"
    build_labels(rp, tp, outp)
    out = pq.read_table(outp).to_pandas().sort_values("receipt_key")
    # Both receipt rows should be labeled 1 and share the same first_valid_txn
    assert set(out["receipt_key"]) == {52, 53}
    assert out["label_same_user_fh"].tolist() == [1, 1]
    assert out["label_same_user_st"].tolist() == [1, 1]
    assert out["label_valid"].tolist() == [1, 1]
    assert out["first_valid_txn_key"].tolist() == [51, 51]

# ----------------------------
# Case 6.2
# Two receipts (same user,coupon); each window has its own txn; each receipt picks its own first_valid
# ----------------------------
def test_two_receipts_two_txns_match_respectively(make_txns, make_receipts,
                                                  add_txn_keys, add_receipt_keys,
                                                  cast_datatype, to_parquet):
    txn = make_txns(
        (6, 9006, "2023-04-02", 2600, 260),
        (6, 9006, "2023-04-05", 2700, 260)
    )
    txn = add_txn_keys(txn, keys=[61, 62])
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_6_2.parquet"; to_parquet(txn, tp)

    # two receipts
    receipt = make_receipts(
        (6, 9006, 260, "2023-04-01", "2023-04-01", "2023-04-06"),
        (6, 9006, 260, "2023-04-04", "2023-04-01", "2023-04-06")
    )
    receipt = add_receipt_keys(receipt, keys=[63, 64])
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_6_2.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_6_2.parquet"
    build_labels(rp, tp, outp)
    out = pq.read_table(outp).to_pandas().sort_values("receipt_key")
    assert list(out["receipt_key"]) == [63, 64]
    assert out["label_same_user_st"].tolist() == [1, 1]
    assert out["label_same_user_fh"].tolist() == [1, 1]
    assert out["label_valid"].tolist() == [1, 1]
    assert out["flag_early"].tolist() == [0, 0]
    assert out["first_valid_txn_key"].tolist() == [61, 62]

# ----------------------------
# Case 7.1
# Cross-user: other uses coupon in owner window; other has NO own covering receipt
# → other_user_in_window_count=1 AND other_user_without_own_receipt_count=1; labels 0
# ----------------------------
def test_cross_user_no_own_receipt(make_txn, make_receipt,
                                   add_txn_key, add_receipt_key,
                                   cast_datatype, to_parquet):
    txn = make_txn(user=202, coupon=9111, pay_date="2023-05-10",
                   pay_amt_cent=2800, reduce_amt_cent=280)
    txn = add_txn_key(txn, key=201)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_7_1.parquet"
    to_parquet(txn, tp)

    owner = make_receipt(user=101, coupon=9111, coupon_amt_cent=280,
                         receive_date="2023-05-01", start_date="2023-05-05", end_date="2023-05-15",
                         price_limit_cent=1000, coupon_status=1)
    owner = add_receipt_key(owner, key=2021)
    owner = cast_datatype(owner, flag="receipt")
    rp = "tests/data_test/receipt_7_1.parquet"; to_parquet(owner, rp)

    outp = "tests/data_test/labels_out_7_1.parquet"
    build_labels(rp, tp, outp)
    row = pq.read_table(outp).to_pandas().iloc[0]
    assert row["label_same_user_fh"] == 0
    assert row["label_same_user_st"] == 0
    assert row["label_valid"] == 0
    assert row["other_user_in_window_txn_count"] == 1
    assert row["other_user_without_own_receipt_txn_count"] == 1
    assert row["flag_cross_user"] == 1

# ----------------------------
# Case 7.2
# Cross-user: other uses coupon in owner window; other DOES have own covering receipt
# → other_user_in_window_count=1 AND other_user_without_own_receipt_count=0; labels 0
# ----------------------------
def test_cross_user_has_own_receipt(make_txn, make_receipt,
                                    add_txn_key, add_receipt_key,
                                    cast_datatype, to_parquet):
    txn = make_txn(user=303, coupon=9222, pay_date="2023-06-20",
                   pay_amt_cent=3000, reduce_amt_cent=300)
    txn = add_txn_key(txn, key=301)
    txn = cast_datatype(txn, flag="txn")
    tp = "tests/data_test/txn_7_2.parquet"; 
    to_parquet(txn, tp)

    # Owner receipt (no same-user redemption)
    owner = make_receipt(user=111, coupon=9222, coupon_amt_cent=300,
                         receive_date="2023-06-01", start_date="2023-06-10", end_date="2023-06-30",
                         price_limit_cent=1000, coupon_status=1)
    owner = add_receipt_key(owner, key=3021)

    # Other user's own covering receipt
    other_r = make_receipt(user=303, coupon=9222, coupon_amt_cent=300,
                           receive_date="2023-06-05", start_date="2023-06-10", end_date="2023-06-30",
                           price_limit_cent=1000, coupon_status=1)
    other_r = add_receipt_key(other_r, key=3022)

    receipt = pd.concat([owner, other_r], ignore_index=True)
    receipt = cast_datatype(receipt, flag="receipt")
    rp = "tests/data_test/receipt_7_2.parquet"; to_parquet(receipt, rp)

    outp = "tests/data_test/labels_out_7_2.parquet"
    build_labels(rp, tp, outp)
    owner_row = pq.read_table(outp).to_pandas().query("receipt_key == 3021").iloc[0]
    assert owner_row["label_same_user_fh"] == 0
    assert owner_row["label_same_user_st"] == 0
    assert owner_row["label_valid"] == 1
    assert owner_row["other_user_in_window_txn_count"] == 1
    assert owner_row["other_user_without_own_receipt_txn_count"] == 0
    assert owner_row["flag_cross_user"] == 0















    