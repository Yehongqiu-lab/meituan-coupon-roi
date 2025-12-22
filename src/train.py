import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from io_load import *
def load_policy_training_3folds_data(repo_root, 
                                     pickle_path="data_work/trainable_colnames.pkl",
                                     train_set_path="meituan-coupon-roi/data_work/policy_train_set_w_CV.parquet"):
    fold_train_1 = [("fold_1_marker", "==", 1)]
    fold_val_1 = [("fold_1_marker", "==", 2)]
    fold_train_2 = [("fold_2_marker", "==", 1)]
    fold_val_2 = [("fold_2_marker", "==", 2)]
    fold_train_3 = [("fold_3_marker", "==", 1)]
    fold_val_3 = [("fold_3_marker", "==", 2)]

    pickle_path = os.path.join(repo_root, pickle_path)

    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)

    policy_cols = list(
        {x for x in cols 
        if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                     "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                     "label_same_user_fh", "label_same_user_st",
                      "Receive_date", "Start_date", "End_date"]})

    policy_train_1 = load_df_from_pq(train_set_path, 
                                 cols=policy_cols, 
                                 filters=fold_train_1)
    policy_val_1 = load_df_from_pq(train_set_path, 
                               cols=policy_cols,
                               filters=fold_val_1)
    policy_train_2 = load_df_from_pq(train_set_path, 
                                 cols=policy_cols, 
                                 filters=fold_train_2)
    policy_val_2 = load_df_from_pq(train_set_path, 
                               cols=policy_cols,
                               filters=fold_val_2)
    policy_train_3 = load_df_from_pq(train_set_path, 
                                 cols=policy_cols, 
                                 filters=fold_train_3)
    policy_val_3 = load_df_from_pq(train_set_path, 
                               cols=policy_cols,
                               filters=fold_val_3)
    return policy_train_1, policy_train_2, policy_train_3, policy_val_1, policy_val_2, policy_val_3

def load_policy_training_data(repo_root, 
                            pickle_path="data_work/trainable_colnames.pkl",
                            train_set_path="meituan-coupon-roi/data_work/policy_train_set.parquet"):
    pickle_path = os.path.join(repo_root, pickle_path)
    
    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)

    policy_cols = list(
        {x for x in cols 
        if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                     "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                     "label_same_user_fh", "label_same_user_st",
                      "Receive_date", "Start_date", "End_date"]})

    policy_train = load_df_from_pq(train_set_path,
                                   cols=policy_cols)
    
    return policy_train

def load_policy_test_data(repo_root,
                          pickle_path="data_work/trainable_colnames.pkl",
                          test_set_path="meituan-coupon-roi/data_work/policy_test_set.parquet"):
    pickle_path = os.path.join(repo_root, pickle_path)
    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)

    policy_cols = list(
        {x for x in cols 
        if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                     "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                     "label_same_user_fh", "label_same_user_st",
                      "Receive_date", "Start_date", "End_date"]})

    policy_test = load_df_from_pq(test_set_path,
                                  cols=policy_cols)
    return policy_test

def get_policy_X_y_data(df):
    y = df["label_invalid"]
    X = df.drop(columns=['label_invalid'])
    return X, y

# TODO: make pickle_path and train_set_path as input arguments
def load_ROI_training_3folds_data(repo_root, fh_or_st):
    fold_train_1 = [[("fold_1_marker", "==", 1), ("label_invalid", "==", 0)]]
    fold_val_1 = [[("fold_1_marker", "==", 2), ("label_invalid", "==", 0)]]
    fold_train_2 = [[("fold_2_marker", "==", 1), ("label_invalid", "==", 0)]]
    fold_val_2 = [[("fold_2_marker", "==", 2), ("label_invalid", "==", 0)]]
    fold_train_3 = [[("fold_3_marker", "==", 1), ("label_invalid", "==", 0)]]
    fold_val_3 = [[("fold_3_marker", "==", 2), ("label_invalid", "==", 0)]]

    pickle_path = os.path.join(repo_root, "data_work/trainable_colnames.pkl")
    train_set_path = "meituan-coupon-roi/data_work/ROI_train_set_w_CV.parquet"

    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)

    if fh_or_st == "fh":
        ROI_cols = list(
            {x for x in cols 
            if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                         "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                         "label_same_user_st", "label_invalid",
                        "Receive_date", "Start_date", "End_date"]})
    elif fh_or_st == "st":
        ROI_cols = list(
            {x for x in cols 
            if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                         "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                         "label_same_user_fh", "label_invalid",
                        "Receive_date", "Start_date", "End_date"]})
    else:
        raise ValueError("fh_or_st must be either 'fh' or 'st'.")

    ROI_train_1 = load_df_from_pq(train_set_path, 
                                 cols=ROI_cols, 
                                 filters=fold_train_1)
    ROI_val_1 = load_df_from_pq(train_set_path, 
                               cols=ROI_cols,
                               filters=fold_val_1)
    ROI_train_2 = load_df_from_pq(train_set_path, 
                                 cols=ROI_cols, 
                                 filters=fold_train_2)
    ROI_val_2 = load_df_from_pq(train_set_path, 
                               cols=ROI_cols,
                               filters=fold_val_2)
    ROI_train_3 = load_df_from_pq(train_set_path, 
                                 cols=ROI_cols, 
                                 filters=fold_train_3)
    ROI_val_3 = load_df_from_pq(train_set_path, 
                               cols=ROI_cols,
                               filters=fold_val_3)
    return ROI_train_1, ROI_train_2, ROI_train_3, ROI_val_1, ROI_val_2, ROI_val_3

def load_ROI_test_data(repo_root, 
                       pickle_path="data_work/trainable_colnames.pkl",
                        test_set_path="meituan-coupon-roi/data_work/ROI_test_set.parquet",
                       fh_or_st="st"):
    pickle_path = os.path.join(repo_root, pickle_path)
    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)

    if fh_or_st == "fh":
        ROI_cols = list(
            {x for x in cols 
            if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                         "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                         "label_invalid", "label_same_user_st",
                      "Receive_date", "Start_date", "End_date"]})
    elif fh_or_st == "st":
        ROI_cols = list(
            {x for x in cols
            if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                         "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                         "label_same_user_fh", "label_invalid",
                        "Receive_date", "Start_date", "End_date"]}
        )
    else:
        raise ValueError("fh_or_st must be either 'fh' or 'st'.")

    ROI_test = load_df_from_pq(test_set_path,
                                  cols=ROI_cols)
    return ROI_test

def get_ROI_X_y_data(df, fh_or_st):
    if fh_or_st == "fh":
        y = df["label_same_user_fh"]
        X = df.drop(columns=['label_same_user_fh'])
    elif fh_or_st == "st":
        y = df["label_same_user_st"]
        X = df.drop(columns=['label_same_user_st'])
    else:
        raise ValueError("fh_or_st must be either 'fh' or 'st'.")
    return X, y

def add_vanilla_weights(df, binary_label_col):
    N = df.shape[0]
    weight = df.groupby(binary_label_col)[binary_label_col].apply(lambda df: N / df.shape[0])
    weight_df = weight.to_frame()
    weight_df.rename(columns={'label_invalid': 'weight'}, inplace=True)
    weight_df = weight_df.reset_index()
    new_df = df.join(weight_df, on=binary_label_col, how='left', lsuffix='_l')
    return new_df['weight']
    
