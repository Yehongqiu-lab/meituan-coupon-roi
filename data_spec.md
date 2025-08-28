# Raw Datasets: 

- **Real world** business data of a cross-sell e-commerce platform from 2023/01/01 to 2023/06/30.
- **De-sensitized:** Contains ~50K user data with no specific user privacy involved.
- Includes transactions details, user log-ins history, and coupon receiving details.

## Keys/Grains: 

- The field name `User_id` is the common key across all 3 tables.
- The field name `Coupon_id` is the key across `order_detail.csv` and `user_coupon_receive.csv`.
- All time-related fields use a single TZ (Asia/Shanghai).
- The meanings of field names are as follows:

In `order_detail.csv`: transaction details of users; short as `txn`.
| Field Name     | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| User_id        | Unique user ID after login                                                  |
| Shop_id        | Merchant ID                                                                 |
| Order_id       | Order ID                                                                    |
| Coupon_id      | Coupon ID used in this order                                                |
| Coupon_type    | Type of coupon used (distinguished by numbers, exact meaning not required)   |
| Biz_code       | Business line code (distinguished by numbers, exact meaning not required)    |
| Pay_date       | User payment date                                                           |
| Actual_pay     | Actual transaction amount paid by user (Order original price − Subsidy)      |
| Reduce_amount  | Subsidy amount                                                              |

In `user_visit_detail.csv`: users' timestamps of log-ins
| Field Name     | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| User_id        | Unique user ID after login                                                  |
| Visit_date     | The date on which users logged in to the platform App or website            |

In `user_coupon_receive.csv`: user history of coupons receipts, and coupon discount details; short as `receipt`.
| Field Name     | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| User_id        | Unique user ID after login                                                  |
| Coupon_id      | Coupon ID                                                                   |
| Coupon_status  | Coupon status (1 = Unused, 2 = Used, 3 = Other)                             |
| Coupon_amt     | Coupon face value (unit: RMB yuan)                                          |
| Receive_time   | Coupon receive time (format: yyyy-MM-dd)                                    |
| Start_time     | Coupon effective start time (format: yyyy-MM-dd)                            |
| End_time       | Coupon expiration time (format: yyyy-MM-dd)                                 |
| Price_limit    | Minimum spending threshold for coupon usage (unit: RMB yuan)                |

## Known quirks:

- **Coupon_id in txns not logged as receipt:** Some users pass coupons on to others.
- **Transactions without Coupon_id:** Two cases — (1) coupon used but missed, inferred via the **Reconciliation rules** introduced below; (2) no coupon applied.
- **Repeated Coupon Use** A user shows multiple transactions with the same Coupon_id despite a one-time receiving record.
- **Repeated Coupon Receipt:** A user receives a same Coupon_id on different days.

# Intermediate Data Processing:

## Data Compression/Type Declare:
- reduce memory by re-declare data type.
- re-align User_id, Coupon_id, Order_id, Shop_id across CSVs.
- re-format date-related fields format.

## Imputation/Reconciliation (txn ↔ receipt): 
Impute missing `Coupon_id` in `txn` with the info from `receipt`.

- **Only** impute `Coupon_id` when **one and only one** same-user receipt matches:
    - `Receive_time ≤ Pay_date ≤ End_time`
- If matched uniquely: set `coupon_id_imputed = 1`.
- Else if none is matched and `Reduce_amount` is 0: set `flag_no_coupon = 1`.
- Else: set `flag_ambiguous_txn = 1`

## Diagnostic Flags  
Flags are retained only for diagnostic purposes (not used in training). 

- **`flag_no_coupon`**: mentioned above.
- **`flag_ambiguous_txn`**: mentioned above.

- **`flag_transfer_case`**: Coupon redemption recorded under a *different* user.  
  - The redeemer did not receive the coupon before the transaction, **or**  
  - The coupon was received but redeemed after its `End_time`.  

- **`flag_untracked_coupon`**: Coupon redemption with **no valid prior receipt**.  
  - Case 1: `txn.Coupon_id` missing → no candidate coupon found by reconciliation, but `Reduce_amount > 0`.  
  - Case 2: `txn.Coupon_id` present in transaction but not found in receipt records.  

- **`flag_repeat_redemption`**: 
    - **each receipt contributes to only one redemption:** if a (user_id, coupon_id) appears in multiple *receipts*, treated as **different** events, and each receipt event matches *at most* one txn and the matched txns must be **different**.
    - **pick the earliest qualifying txn:** if a (user_id, coupon_id) appears in multiple *txns*, only the earliest qualifying txn can be considered the redemption used to create labels; later ones are flagged 1 in `flag_repeat_redemption`.

- **`flag_pay_void`**:
    - `Actual_pay ≤ 0` in txns.

## Data Quality / Integrity Metrics  

### Receipt-Level Diagnostics 

For each segment defined by `Price_limit_bin × Coupon_amt_bin × Expiry_span_bin`:  
- Report: **% `flag_transfer_case`** = (# transfer_case) / (# receipts).  
- **High-integrity cohort** = segments with low % `flag_transfer_case`.  

### Transaction-Level Diagnostics  

Across all transactions, report the proportions of:  
- `coupon_id_imputed = 1`  
- `flag_ambiguous_txn`  
- `flag_no_coupon`  
- `flag_untracked_coupon`
- `flag_repeat_redemption`
- `flag_pay_void`

## Supervised Learning Targets:
Generate training labels from two perspectives: short-term and long-term.

For each receipt,
- **Short-term ROI:** `label_same_user_st` (primary label): 1 if **same user** redeems within `min(15d, End_time)`; else 0.
- **Full-horizon ROI:** `label_same_user_fh`: if **same user** redeems within `End_time`.

## Leakage Guards:

- All features must satisfy `feature_ts ≤ Receive_time`.
- ML Workflow Data **Split key** = `Receive_time`.
- **Purge Period:** apply a 15-day gap before each Val/Test boundary so no Train label window crosses into future data.
- Any population aggregates (e.g., redemption rate by coupon_type) must be computed within **feature lookback window only**.

## Edge Cases/Drop Rules:

- Duplicate rows in any CSV.
- Receipts with `Start_time` after `End_time`.
- Receipts with missing `Start_time/End_time`.

# Processed Dataset:

## Entities:
Row = one coupon receipt event for a (User_id, Coupon_id, Receive_time).

## Label:

Choose one of the following labels aligning to the product design:
- `label_same_user_st`
- `label_same_user_fh`

## Split:

- **Split key:** `Receive_time` of coupons
- **Split integrity:** Leave a 15-day purge before Test:  
    - Train: 2023-01-01 … 2023-05-16 (4.5 months).
    - Test:  2023-06-01 … 2023-06-30 (1 month).
- **3-Fold Forward-Chaining CV:** used within Train with 15-day purge before Val:
    - Fold 1: Train: 2023-01-01 … 2023-02-28, Val: 2023-03-16 … 2023-03-31.
    - Fold 2: Train: 2023-01-01 … 2023-03-31, Val: 2023-04-16 … 2023-04-30.
    - Fold 3: Train: 2023-01-01 … 2023-04-30, Val: 2023-05-16 … 2023-05-31.

## Features:

- **User-centric features:**
    - **As-of features only:** features timestamp `≤ Receive_time`.
    - **Per-fold fitting:** scalers/encoders/samplers fit on the Train-fold only.
    - **Examples:** user behavior(feature lookback window: 7/30/90d)
- **Coupon-centric features:**
    - **Examples:** coupon details(face value, min spend, subsidy)



