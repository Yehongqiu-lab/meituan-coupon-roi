Keys/grain: user_id, coupon_id, receive_ts, redeem_ts, txn_ts

Label: redeemed within min(15d, expiry) from receive_ts

Split: time-based train/val/test by month

Leakage rule: features strictly pre-receive_ts
