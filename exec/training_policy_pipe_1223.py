import json
import pickle
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import os
import sys
root = os.path.dirname(os.getcwd())
repo_root = os.path.join(root, "meituan-coupon-roi")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.train import *
date_p = "1217"
date = "1223"
m_id = "1"

params_yaml_path = os.path.join(repo_root, f"conf/params_policy_{date_p}.yaml")
model_saving_path_1 = os.path.join(repo_root, f"models/policy_{date}_{m_id}_fd1.txt")
model_saving_path_2 = os.path.join(repo_root, f"models/policy_{date}_{m_id}_fd2.txt")
model_saving_path_3 = os.path.join(repo_root, f"models/policy_{date}_{m_id}_fd3.txt")

print("Loading data for policy training...")
policy_train_1, policy_train_2, policy_train_3, policy_val_1, policy_val_2, policy_val_3 = \
    load_policy_training_3folds_data(repo_root)


X_train_1, y_train_1 = get_policy_X_y_data(policy_train_1)
X_train_2, y_train_2 = get_policy_X_y_data(policy_train_2)
X_train_3, y_train_3 = get_policy_X_y_data(policy_train_3)

W_train_1 = add_vanilla_weights(policy_train_1, "label_invalid")
W_train_2 = add_vanilla_weights(policy_train_2, "label_invalid")
W_train_3 = add_vanilla_weights(policy_train_3, "label_invalid")

X_val_1, y_val_1 = get_policy_X_y_data(policy_val_1)
X_val_2, y_val_2 = get_policy_X_y_data(policy_val_2)
X_val_3, y_val_3 = get_policy_X_y_data(policy_val_3)

W_val_1 = add_vanilla_weights(policy_val_1, "label_invalid")
W_val_2 = add_vanilla_weights(policy_val_2, "label_invalid")
W_val_3 = add_vanilla_weights(policy_val_3, "label_invalid")


lgb_train_1 = lgb.Dataset(
    X_train_1, y_train_1, 
    weight=W_train_1,
    feature_name=X_train_1.columns.to_list(),
    categorical_feature=["Price_limit_bin", "Coupon_limit_bin", "Expiry_span_bin"],
    free_raw_data=True
)
lgb_val_1 = lgb.Dataset(
    X_val_1, y_val_1, reference=lgb_train_1, weight=W_val_1, free_raw_data=True
)

lgb_train_2 = lgb.Dataset(
    X_train_2, y_train_2,
    weight=W_train_2, 
    feature_name=X_train_2.columns.to_list(),
    categorical_feature=["Price_limit_bin", "Coupon_limit_bin", "Expiry_span_bin"],
    free_raw_data=True
)
lgb_val_2 = lgb.Dataset(
    X_val_2, y_val_2, reference=lgb_train_2, weight=W_val_2, free_raw_data=True
)

lgb_train_3 = lgb.Dataset(
    X_train_3, y_train_3, 
    weight=W_train_3,
    feature_name=X_train_3.columns.to_list(),
    categorical_feature=["Price_limit_bin", "Coupon_limit_bin", "Expiry_span_bin"],
    free_raw_data=True
)
lgb_val_3 = lgb.Dataset(
    X_val_3, y_val_3, reference=lgb_train_3, weight=W_val_3, free_raw_data=True
)



with open(params_yaml_path, 'r') as f:
    params = yaml.safe_load(f)

print("starting policy training with cross validation...")

gbm_1 = lgb.train(
    params,
    lgb_train_1,
    num_boost_round=10,
    valid_sets=lgb_train_1
)
auc_1 = roc_auc_score(y_train_1, gbm_1.predict(X_train_1))
gbm_1.save_model(model_saving_path_1)

gbm_2 = lgb.train(
    params,
    lgb_train_2,
    num_boost_round=10,
    valid_sets=lgb_train_2
)
auc_2 = roc_auc_score(y_train_2, gbm_2.predict(X_train_2))
gbm_2.save_model(model_saving_path_2)

gbm_3 = lgb.train(
    params,
    lgb_train_3,
    num_boost_round=10,
    valid_sets=lgb_train_3
)
auc_3 = roc_auc_score(y_train_3, gbm_3.predict(X_train_3))
gbm_3.save_model(model_saving_path_3)

auc_in_sample = (auc_1 + auc_2 + auc_3) / 3
print("The in-sample auc score is:", auc_in_sample)

"""
print("start policy training based on all available training data...")
policy_train = load_policy_training_data(repo_root)
policy_test = load_policy_test_data(repo_root)

X_train, y_train = get_policy_X_y_data(policy_train)
W_train = add_vanilla_weights(policy_train, "label_invalid")

X_test, y_test = get_policy_X_y_data(policy_test)
W_test = add_vanilla_weights(policy_test, "label_invalid")

# TODO: finish the train dataset below:
lgb_train = lgb.Dataset()

lgb_eval = lgb.Dataset(
    X_test, y_test, reference=lgb_train_1, weight=W_test, free_raw_data=True
)
"""