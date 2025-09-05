# tests/test_reconsile.py
import pandas as pd
from src.reconcile import impute_missing_coupon_ids

def test_happy_no_need_to_impute(make_txn, make_receipt):
    """
    Case 0: No txns with missing Coupon_id_code
    Expect:
      - input txns_df is returned unchanged, with added columns
      - coupon_id_imputed == 0
      - all other flags == 0
    """
    # Given: one txn with Coupon_id_code present, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=100, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one matching receipt for the same user with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-09",
        end_date="2023-01-15",
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, unchanged except added columns
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == 100, "Expected Coupon_id_code to remain unchanged"

    # flags are clean
    for flag in [
        "flag_no_coupon", 
        "flag_ambiguous_txn"
    ]:
        assert flag in out.columns, f"Output must expose flag {flag}"
        assert row[flag] == 0, f"Expected flag {flag} to be 0"



def test_happy_path_single_valid_receipt(make_txn, make_receipt):
    """
    Case 1: Happy path — one valid receipt uniquely covers Pay_date
    Expect:
      - Coupon_id is imputed
      - coupon_id_imputed == 1
      - all diagnostic flags == 0
    """
    # Given: one txn with Coupon_id_code==-1(indicates missing in raw data), positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one matching receipt for the same user with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-09",
        end_date="2023-01-15",
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with imputed coupon and clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # Imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 1, "Expected coupon to be imputed"

    # Coupon_id_code is filled in
    assert row["Coupon_id_code"] == 100, "Expected Coupon_id_code to be imputed from receipt"

    # flags are clean
    for flag in [
        "flag_no_coupon", 
        "flag_ambiguous_txn"
    ]:
        assert flag in out.columns, f"Output must expose flag {flag}"
        assert row[flag] == 0, f"Expected flag {flag} to be 0"
    
def test_no_match_zero_reduce(make_txn, make_receipt):
    """
    Case 2: No matching receipt covering Pay_date, and zero reduce amount in txn
    Expect:
      - coupon_id_imputed == 0
      - flag_no_coupon == 1
      - flag_ambiguous_txn == 0
    """
    # Given: one txn with missing Coupon_id_code, positive pay, zero reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=0)
    
    # And: one non-matching receipt for the same user (window does not cover Pay_date)
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-06",
        end_date="2023-01-08",  # window ends before Pay_date
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 1, "Expected flag_no_coupon to be 1"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 0, "Expected flag_ambiguous_txn to be 0"

def test_no_match_positive_reduce(make_txn, make_receipt):
    """
    Case 3: No matching receipt covering Pay_date, and positive reduce amount in txn
    Expect:
      - coupon_id_imputed == 0
      - flag_no_coupon == 0
      - flag_ambiguous_txn == 1
    """
    # Given: one txn with missing Coupon_id_code, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one non-matching receipt for the same user (window does not cover Pay_date)
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-06",
        end_date="2023-01-08",  # window ends before Pay_date
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 0, "Expected flag_no_coupon to be 0"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 1, "Expected flag_ambiguous_txn to be 1"

def test_multiple_candidate_receipts(make_txn, make_receipt):
    """
    Case 4: Multiple candidate receipts covering Pay_date
    Expect:
      - coupon_id_imputed == 0
      - flag_no_coupon == 0
      - flag_ambiguous_txn == 1
    """
    # Given: one txn with missing Coupon_id_code, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: two matching receipts for the same user with active windows covering Pay_date
    receipt = pd.concat([
        make_receipt(
            user=1,
            coupon=100,
            coupon_amt_cent=500,
            receive_date="2023-01-05",
            start_date="2023-01-09",
            end_date="2023-01-15",
            price_limit_cent=1000,
            coupon_status=1),
        make_receipt(
            user=1,
            coupon=101,
            coupon_amt_cent=300,
            receive_date="2023-01-07",
            start_date="2023-01-10",
            end_date="2023-01-20",
            price_limit_cent=800,
            coupon_status=1)
    ], ignore_index=True)
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 0, "Expected flag_no_coupon to be 0"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 1, "Expected flag_ambiguous_txn to be 1"

def test_boundary_inclusivity(make_txn, make_receipt):
    """
    Case 5: Boundary inclusivity — receipt window exactly matches Pay_date
    Expect:
      - Coupon_id is imputed
      - coupon_id_imputed == 1
      - all other diagnostic flags == 0
    """
    # Given: one txn with missing Coupon_id_code, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one matching receipt for the same user with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-10",  # start_date == Pay_date
        end_date="2023-01-10",    # end_date == Pay_date
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with imputed coupon and clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # Imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 1, "Expected coupon to be imputed"

    # Coupon_id_code is filled in
    assert row["Coupon_id_code"] == 100, "Expected Coupon_id_code to be imputed from receipt"

    # flags are clean
    for flag in [
        "flag_no_coupon", 
        "flag_ambiguous_txn"
    ]:
        assert flag in out.columns, f"Output must expose flag {flag}"
        assert row[flag] == 0, f"Expected flag {flag} to be 0"

def test_receive_date_after_start_date(make_txn, make_receipt):
    """
    Case 6: Receipt with Receive_date after Start_date is still valid
    Expect:
      - Coupon_id is imputed
      - coupon_id_imputed == 1
      - all other diagnostic flags == 0
    """
    # Given: one txn with missing Coupon_id_code, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one matching receipt for the same user with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-09",  # Receive_date after Start_date
        start_date="2023-01-08",
        end_date="2023-01-15",
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with imputed coupon and clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # Imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 1, "Expected coupon to be imputed"

    # Coupon_id_code is filled in
    assert row["Coupon_id_code"] == 100, "Expected Coupon_id_code to be imputed from receipt"

    # flags are clean
    for flag in [
        "flag_no_coupon", 
        "flag_ambiguous_txn"
    ]:
        assert flag in out.columns, f"Output must expose flag {flag}"
        assert row[flag] == 0, f"Expected flag {flag} to be 0"

def test_bad_receipt_span_gets_dropped(make_txn, make_receipt):
    """
    Case 7: Receipt with invalid span (Start_date > End_date) is dropped
    Expect:
      - coupon_id_imputed == 0
      - flag_no_coupon == 0
      - flag_ambiguous_txn == 1 because reduce_amt_cent > 0
    """
    # Given: one txn with missing Coupon_id_code, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one non-matching receipt for the same user (invalid window)
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-12",  # Start_date after End_date
        end_date="2023-01-08",
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 0, "Expected flag_no_coupon to be 0"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 1, "Expected flag_ambiguous_txn to be 1"

def test_receipts_with_missing_User_id_gets_dropped(make_txn, make_receipt):
    """
    Case 8: Receipts with missing User_id_code (-1) are dropped.
    Expect:
        - coupon_id_imputed == 0
        - flag_no_coupon == 1 because Reduce_amount_cent == 0 in txn
        - flag_ambiguous_txn == 0
    """
    # Given: one txn with missing Coupon_id_code, positive pay, zero reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=0)
    
    # And: one non-matching receipt with missing User_id_code
    receipt = make_receipt(
        user=-1,  # missing User_id_code
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-06",
        end_date="2023-01-12",  # window covers Pay_date
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 1, "Expected flag_no_coupon to be 1"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 0, "Expected flag_ambiguous_txn to be 0"

def test_different_user_is_ignored(make_txn, make_receipt):
    """
    Case 9: Receipt for a different user is ignored.
    Expect:
        - coupon_id_imputed == 0
        - flag_no_coupon == 1 because Reduce_amount_cent == 0 in txn
        - flag_ambiguous_txn == 0
    """
    # Given: one txn with missing Coupon_id_code, positive pay, zero reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=0)
    
    # And: one non-matching receipt for a different user
    receipt = make_receipt(
        user=2,  # different user
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-06",
        end_date="2023-01-12",  # window covers Pay_date
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 1, "Expected flag_no_coupon to be 1"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 0, "Expected flag_ambiguous_txn to be 0"

def test_txn_with_incomplete_User_id_is_ambiguous(make_txn, make_receipt):
    """
    Case 10: Txn with missing User_id_code or Pay_date is treated as ambiguous
    Expect:
      - coupon_id_imputed == 0
      - flag_no_coupon == 0
      - flag_ambiguous_txn == 1
    """
    # Given: one txn with missing User_id_code, positive pay, positive reduce;
    # the other with missing Pay_date is covered in test case 11:
    txn = make_txn(
        user=-1,  # missing User_id_code
        coupon=-1, 
        pay_date="2023-01-10",
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one matching receipt for user=1 with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-09",
        end_date="2023-01-15",
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 0, "Expected flag_no_coupon to be 0"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 1, "Expected flag_ambiguous_txn to be 1"

def test_txn_with_incomplete_Pay_date_is_ambiguous(make_txn, make_receipt):
    """
    Case 11: Txn with missing Pay_date is treated as ambiguous
    Expect:
      - coupon_id_imputed == 0
      - flag_no_coupon == 0
      - flag_ambiguous_txn == 1
    """
    # Given: one txn with missing Pay_date, positive pay, positive reduce:
    txn = make_txn(
        user=1, 
        coupon=-1, 
        pay_date=None,  # missing Pay_date
        pay_amt_cent=5000,
        reduce_amt_cent=500)
    
    # And: one matching receipt for user=1 with an active window covering Pay_date
    receipt = make_receipt(
        user=1,
        coupon=100,
        coupon_amt_cent=500,
        receive_date="2023-01-05",
        start_date="2023-01-09",
        end_date="2023-01-15",
        price_limit_cent=1000,
        coupon_status=1)    
    
    # When:
    out = impute_missing_coupon_ids(txn, receipt)

    # Then: one row back, with clean flags
    assert len(out) == 1, "Should return exactly one row"
    row = out.iloc[0]

    # No imputation happened
    assert "coupon_id_imputed" in out.columns, "Output must expose imputation flag"
    assert row["coupon_id_imputed"] == 0, "Expected coupon NOT to be imputed"
    assert row["Coupon_id_code"] == -1, "Expected Coupon_id_code to remain missing"

    # flags are as expected
    assert "flag_no_coupon" in out.columns, "Output must expose flag flag_no_coupon"
    assert row["flag_no_coupon"] == 0, "Expected flag_no_coupon to be 0"

    assert "flag_ambiguous_txn" in out.columns, "Output must expose flag flag_ambiguous_txn"
    assert row["flag_ambiguous_txn"] == 1, "Expected flag_ambiguous_txn to be 1"