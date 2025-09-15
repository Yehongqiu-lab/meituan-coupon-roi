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

In `user_visit_detail.csv`: users' timestamps of log-ins; short as `user_visit`.
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
| Receive_date   | Coupon receive time (format: yyyy-MM-dd)                                    |
| Start_date     | Coupon effective start time (format: yyyy-MM-dd)                            |
| End_date       | Coupon expiration time (format: yyyy-MM-dd)                                 |
| Price_limit    | Minimum spending threshold for coupon usage (unit: RMB yuan)                |

## Known quirks:

- **Coupon_id in txns not logged as receipt:** Some users pass coupons on to others.
- **Transactions without Coupon_id:** Two cases — (1) coupon used but missed, inferred via the **Reconciliation rules** introduced below; (2) no coupon applied.
- **Repeated Coupon Use** A user shows multiple transactions with the same Coupon_id despite a one-time receiving record.
- **Repeated Coupon Receipt:** A user receives a same Coupon_id on different days.
- **Actual benefit differs from what's stated:** When a coupon is actually redeemed, the actual benefit/reduced amount might be different from what's stated in receipt. It differs based on situations.

# Intermediate Data Processing:

## Data Compression/Type Declare:
- reduce memory by re-declare data type.
- re-align User_id, Coupon_id, Order_id, Shop_id across CSVs.
- re-format date-related fields.

## Imputation/Reconciliation (txn ↔ receipt): 
Impute missing `txn.Coupon_id` with the info from `receipt`.

- **Only** impute `Coupon_id` when **one and only one** same-user receipt matches:
    - `max(Receive_date, Start_date) ≤ Pay_date ≤ End_date`
- If matched uniquely: set `coupon_id_imputed = 1`.
- Else if none is matched and `Reduce_amount` is 0: set `flag_no_coupon = 1`.
- Else: set `flag_ambiguous_txn = 1`

## Diagnostic Flags  
Flags are retained only for diagnostic purposes (not used in training). 

### The following flags are labeled within the `txn` table:

- Flags ONLY for txns with missing Coupon_id in raw data:
    - **`flag_no_coupon`**: mentioned above.
    - **`flag_ambiguous_txn`**: mentioned above.

- **`flag_untracked_coupon`**: Coupon redemption with **no valid prior receipt**.  
    - `txn.Coupon_id` present in transaction but not found in any users' receipt records. 
    - excludes the txns with the Coupon_id still missing.

- **`flag_missing_info`**:
    - Any fields missing in txns.
    - Coupon_id not counted as missing if `flag_no_coupon` is 1.

- **`flag_pay_or_reduce_amt_abn`**:
    - `Actual_pay < 0` or `Reduce_amount < 0` in txns. 
    - **Or** `Reduce_amount > 0` when `flag_no_coupon` is 1.

### The following flags labeled within the `receipt` table:

The following 4 flags `flag_struc_invalid`, `flag_cross_user`, `flag_early` and `flag_late` corresponds to 4 main types of invalid usage cases, all **serving the same purpose** of deciding whether a coupon usage is valid or not. 
For invalid usage cases, at least one of these flags is 1.

**Invalid Cases:** The Coupon receipt either has missing info (structural issue), or counterfaits with what's recorded in txns (semantic issue). 
    - **`flag_struc_invalid`:** *Structrual Issue:* The coupon was received by the same user but its `Start_date`, `Receive_date`, or `End_date` is missing, or `Start_date > End_date`.
    - *Semantic Issue:*
        - **`flag_cross_user`** The coupon was redeemed by a different user. The redeemer did not receive the coupon within its valid usage window, **or**
        - **`flag_early` and `flag_late`** The coupon was received by the same user but redeemed before its `Start_date`, or after its `End_date`.
    
## Data Quality / Integrity Metrics  

### Receipt-Level Diagnostics 

For each segment defined by `Price_limit_bin × Coupon_amt_bin × Expiry_span_bin`:  
- Report: 
    - **%`flag_struc_invalid`, `flag_cross_user`, `flag_early` and `flag_late`** = (# invalid_coupon in each type of case) / (# receipts).  
    - **redemption rate** = (# validly_redeemed_coupon) / (# receipts).
    - **repeatedly redeemed rate** = (# repeatedly_redeemed_coupon) / (# receipts)
- **High-integrity cohort** = segments with low % invalid_coupon in each type of case, high redemption rate, and low repeatedly redeemed rate. 

>   Note: **repeat_redemption**: 
        - each receipt contributes to only one redemption: if a (user_id, coupon_id) appears in multiple receipts, treated as different events, and each receipt event matches at most one txn in which the Pay_date covers the valid usage window.

        - pick the earliest qualifying txn: if a (user_id, coupon_id) appears in multiple txns, only the earliest qualifying txn can be considered the redemption used to create labels; later ones are not.

### Transaction-Level Diagnostics  

Across all transactions, report the proportions of:  
- `coupon_id_imputed = 1`, `flag_ambiguous_txn`, and `flag_no_coupon` in txns with Coupon_id oringinally missing.
- txns with missing Coupon_id in the raw data.  
- `flag_untracked_coupon`.
- `flag_missing_info`
- `flag_pay_or_reduce_amt_abn`.

## Supervised Learning Targets:
Generate training labels from two perspectives: 
1. policy/audit: decide if a coupon is used in a valid way or not.
2. redemption/ROI: further divide into two variants: short-term and full-horizon.

### Policy/Audit:
`label_valid`: For each **coupon receipt**, any one of the flags `flag_struc_invalid`, `flag_cross_user`, `flag_early` and `flag_late` is 1, label as 0(invalid); else 1.

### Redemption/ROI:
For each **coupon receipt**,
- **Short-term ROI:** `label_same_user_st` (primary label): 1 if **same user** redeems within `min(15d, End_date)`; else 0.
- **Full-horizon ROI:** `label_same_user_fh`: if **same user** redeems within `End_date`.

Pick one aligning to the ROI campaign.

## Leakage Guards:

- All features must satisfy `feature_ts ≤ Receive_date`.
- Any population aggregates (e.g., redemption rate by coupon_type) must be computed within **feature lookback window only**.
- Train/Val/Test **Split key** = `Receive_date`.
- **Purge Period/Lookahead Guard:** 
    - Right-cencsored data awareness: For receipts whose End_date go beyond the data coverage end (2023/06/30), simply drop these receipts.
    - For validity/policy modeling, apply a label-lookahead guard ensuring all receipts have `End_date ≤ split_start_next - 1d`.
    - For redemption/ROI modeling, apply a 15-day gap before each Val/Test boundary so no Train label window crosses into future data.
    
    
## Edge Cases/Drop Rules:

- Duplicate rows in any CSV.

# Processed Dataset:

## Entities:
Row = one coupon receipt event for a (User_id, Coupon_id, Receive_date).

## Labels:
1. **Policy:** `label_valid`
2. **ROI:** Choose one of the following labels aligning to the product design:
- `label_same_user_st`
- `label_same_user_fh`

## Split:
### Right-censored data handling:
Drop the receipts whose End_date go beyong 2023-06-30.

### Policy modeling:
- **Split key:** `Receive_date` of coupons
- **Split integrity:** Apply a label-lookahead guard before each of the next set's starting timestamp.
    - `End_date ≤ split_start_next - 1d`

### ROI modeling: 
- **Split key:** `Receive_date` of coupons
- **Split integrity:** Leave a 15-day purge before Test:  
    - Train: 2023-01-01 … 2023-05-16 (4.5 months).
    - Test:  2023-06-01 … 2023-06-30 (1 month).
- **3-Fold Forward-Chaining CV:** used within Train with 15-day purge before Val:
    - Fold 1: Train: 2023-01-01 … 2023-02-28, Val: 2023-03-16 … 2023-03-31.
    - Fold 2: Train: 2023-01-01 … 2023-03-31, Val: 2023-04-16 … 2023-04-30.
    - Fold 3: Train: 2023-01-01 … 2023-04-30, Val: 2023-05-16 … 2023-05-31.

## Features:

- **User-centric features:**
    - **As-of features only:** features timestamp `≤ Receive_date`.
    - **Per-fold fitting:** scalers/encoders/samplers fit on the Train-fold only.
    - **Examples:** user behavior(feature lookback window: 7/30/90d)
- **Coupon-centric features:**
    - **Examples:** coupon details(face value, min spend, subsidy)



