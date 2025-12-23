import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from src.io_load import *

def _edit_filter(filter: list[list], arg: tuple) -> list[list]:
    new_filter = []
    for f in filter:
        f.append(arg)
        new_filter.append(f)
    return new_filter

def load_policy_training_3folds_data(repo_root, 
                                     pickle_path="data_work/trainable_colnames.pkl",
                                     train_set_path="meituan-coupon-roi/data_work/policy_train_set_w_CV.parquet",
                                     segment=False,
                                     **kwargs):
    
    fold_train_1 = ("fold_1_marker", "==", 1)
    fold_val_1 = ("fold_1_marker", "!=", 1)
    fold_train_2 = ("fold_2_marker", "==", 1)
    fold_val_2 = ("fold_2_marker", "!=", 1)
    fold_train_3 = ("fold_3_marker", "==", 1)
    fold_val_3 = ("fold_3_marker", "!=", 1)

    pickle_path = os.path.join(repo_root, pickle_path)

    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)
    
    if segment == False:
        policy_cols = list(
        {x for x in cols 
        if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                     "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                     "label_same_user_fh", "label_same_user_st",
                      "Receive_date", "Start_date", "End_date"]})
        filters = [[]]

    else:
        policy_cols, filters = load_policy_certain_segment_data(repo_root,
                                                                **kwargs,
                                                                pickle_path=pickle_path,
                                                                train_set_path=train_set_path,
                                                                w_3folds=True)
    
    fold_train_1 = _edit_filter(filters, fold_train_1)
    fold_val_1 = _edit_filter(filters, fold_val_1)
    fold_train_2 = _edit_filter(filters, fold_train_2)
    fold_val_2 = _edit_filter(filters, fold_val_2)
    fold_train_3 = _edit_filter(filters, fold_train_3)
    fold_val_3 = _edit_filter(filters, fold_val_3)
            
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

def load_policy_certain_segment_data(repo_root,
                                      Price_limit_bin=2,
                                      Coupon_limit_bin=0,
                                      Expiry_span_bin=0,
                                      pickle_path="data_work/trainable_colnames.pkl",
                                      train_set_path="meituan-coupon-roi/data_work/policy_train_set.parquet",
                                      w_3folds=False):
    pickle_path = os.path.join(repo_root, pickle_path)
    
    with open(pickle_path, 'rb') as f:
        cols = pickle.load(f)

    if isinstance(Price_limit_bin, list):
        if isinstance(Coupon_limit_bin, list) and isinstance(Expiry_span_bin, list):
            if len(Price_limit_bin) == len(Coupon_limit_bin) and len(Coupon_limit_bin) == len(Expiry_span_bin):
                policy_cols = list(
                {x for x in cols 
                if x not in ["receipt_key", "receipt_key_1", "receipt_key_2",
                            "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                            "label_same_user_fh", "label_same_user_st",
                            "Receive_date", "Start_date", "End_date"]})
                filters = []
                n = len(Price_limit_bin)
                for i in range(n):
                    filters.append([("Price_limit_bin", "==", Price_limit_bin[i]),
                                    ("Coupon_limit_bin", "==", Coupon_limit_bin[i]),
                                    ("Expiry_span_bin", "==", Expiry_span_bin[i])])
            else:
                raise ValueError("The sizes of the inputted segment identifiers are not equal!")
        else:
            raise TypeError("The type of segment identifiers should be the same!")
    else:
        policy_cols = list(
        {x for x in cols 
        if x not in ["Price_limit_bin", "Coupon_limit_bin", "Expiry_span_bin",
                     "receipt_key", "receipt_key_1", "receipt_key_2",
                     "fold_1_marker", "fold_2_marker", "fold_3_marker", 
                     "label_same_user_fh", "label_same_user_st",
                      "Receive_date", "Start_date", "End_date"]})
        
        filters = [[("Price_limit_bin", "==", Price_limit_bin),
                ("Coupon_limit_bin", "==", Coupon_limit_bin),
                ("Expiry_span_bin", "==", Expiry_span_bin)]]
    
    if w_3folds == False:
        policy_segment_data = load_df_from_pq(train_set_path,
                                           cols=policy_cols,
                                           filters=filters)
        return policy_segment_data
    else:
        return policy_cols, filters

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
    fold_val_1 = [[("fold_1_marker", "!=", 1), ("label_invalid", "==", 0)]]
    fold_train_2 = [[("fold_2_marker", "==", 1), ("label_invalid", "==", 0)]]
    fold_val_2 = [[("fold_2_marker", "!=", 1), ("label_invalid", "==", 0)]]
    fold_train_3 = [[("fold_3_marker", "==", 1), ("label_invalid", "==", 0)]]
    fold_val_3 = [[("fold_3_marker", "!=", 1), ("label_invalid", "==", 0)]]

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
    weight_df.rename(columns={binary_label_col: 'weight'}, inplace=True)
    weight_df = weight_df.reset_index()
    new_df = df.join(weight_df, on=binary_label_col, how='left', lsuffix='_l')
    return new_df['weight']

def metric_individual_class_accuracy(y_true, y_pred, which_class=1, thresholds=[0.25, 0.5, 0.75], test_mode=False):
    """
    :param which_class: 1 returns the recall rate; 0 returns true negative / (true negative + false positive).
    :param thresholds: classification decision rule thresholds
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy(dtype="int")
    elif isinstance(y_true, list):
        y_true = np.array(y_true, dtype="int")
    elif isinstance(y_true, np.ndarray):
        y_true = y_true.astype(int)
    else:
        raise TypeError("Unsupported input type for y_true!")
    
    if not isinstance(y_pred, np.ndarray) and not isinstance(y_pred, list):
        raise TypeError("Unsupported input type for y_pred!")
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred, dtype="float64")
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred should have the same size!")
    n = y_true.shape[0]
    
    output = []
    for th in thresholds:
        d = np.zeros(n, dtype="int")
        denom, numer = 0, 0

        for i in range(n):
            if y_pred[i] >= th:
                d[i] = 1
            if y_true[i] == which_class:
                denom += 1
                if d[i] == which_class:
                    numer += 1

        if denom == 0:
            op = 0
        else:
            op = numer/denom if test_mode else np.round(numer/denom, 3)
        output.append(op)
    
    return output



    

    


