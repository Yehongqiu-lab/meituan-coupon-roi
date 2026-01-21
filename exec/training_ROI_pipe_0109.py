import pickle
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

import os
import sys
root = os.path.dirname(os.getcwd())
repo_root = os.path.join(root, "meituan-coupon-roi")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from train import *
from io_load import *
id_param = "1217"

date = "0110"
m_id = "m_7d"

seg_ids = ["seg_001", "seg_011_and_021", "seg_010", "seg_000", "oth_segs"]
seg_params = []
for p in range(3):
    for c in range(3):
        for e in range(2):
            if [p, c, e] not in [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 2, 1], [0, 0, 0]]:
                seg_params.append([p, c, e])
segs = {"seg_001": [[0, 0, 1]],
        "seg_011_and_021": [[0, 1, 1], [0, 2, 1]],
        "seg_010": [[0, 1, 0]],
        "seg_000": [[0, 0, 0]],
        "oth_segs": seg_params}

params_yaml_path = os.path.join(repo_root, f"conf/params_ROI_{id_param}.yaml")
selected_cols_pkl = f"conf/feature_selection/ROI_modeling/ROI_train_cols_{m_id}.pkl"
test_data = "meituan-coupon-roi/data_work/processed/ROI_test_set_w_netprofit.parquet"

perf_saving_path = os.path.join(repo_root, f"experiment_results/ROI/perf_ROI_{date}_{m_id}.txt")
test_pred_saving_path = os.path.join(repo_root, f"experiment_results/ROI/test_pred_ROI_{date}_{m_id}.csv")
models_saving_paths = {}
for seg_id in seg_ids:
    models_saving_paths[seg_id] = os.path.join(repo_root, f"models/ROI/ROI_{date}_{m_id}_{seg_id}.pkl")

for seg_id in seg_ids:
    print(f"Loading data for ROI training on segment {seg_id}...")
    ROI_train = load_ROI_TrainorTest_data(repo_root,
                                    pickle_path=selected_cols_pkl,
                                    segment=True,
                                    Price_limit_bin=segs[seg_id][0][0],
                                    Coupon_limit_bin=segs[seg_id][0][1],
                                    Expiry_span_bin=segs[seg_id][0][2])
    ROI_test = load_ROI_TrainorTest_data(repo_root,
                                    pickle_path=selected_cols_pkl,
                                    data_set_path=test_data,
                                    segment=True,
                                    Price_limit_bin=segs[seg_id][0][0],
                                    Coupon_limit_bin=segs[seg_id][0][1],
                                    Expiry_span_bin=segs[seg_id][0][2])
    ROI_test_bus_col = load_df_from_pq(test_data,
                                       cols=["Actual_pay_cent"],
                                       filters=[[("Price_limit_bin", "==", segs[seg_id][0][0]),
                                                ("Coupon_limit_bin", "==", segs[seg_id][0][1]),
                                                ("Expiry_span_bin", "==", segs[seg_id][0][2])]])["Actual_pay_cent"]
    
    if len(segs[seg_id]) > 1:
        for seg_param in segs[seg_id][1:]:
            train = load_ROI_TrainorTest_data(repo_root,
                                        pickle_path=selected_cols_pkl,
                                        segment=True,
                                        Price_limit_bin=seg_param[0],
                                        Coupon_limit_bin=seg_param[1],
                                        Expiry_span_bin=seg_param[2])
            test = load_ROI_TrainorTest_data(repo_root,
                                        pickle_path=selected_cols_pkl,
                                        data_set_path=test_data,
                                        segment=True,
                                        Price_limit_bin=seg_param[0],
                                        Coupon_limit_bin=seg_param[1],
                                        Expiry_span_bin=seg_param[2]
                                        )
            test_bus_col = load_df_from_pq(test_data, 
                                           cols=["Actual_pay_cent"],
                                           filters=[[("Price_limit_bin", "==", seg_param[0]),
                                                     ("Coupon_limit_bin", "==", seg_param[1]),
                                                      ("Expiry_span_bin", "==", seg_param[2])]])["Actual_pay_cent"]
            
            ROI_train = pd.concat([ROI_train, train], ignore_index=True)
            ROI_test = pd.concat([ROI_test, test], ignore_index=True)
            ROI_test_bus_col = pd.concat([ROI_test_bus_col, test_bus_col], ignore_index=True)

    X_train, y_train = get_ROI_X_y_data(ROI_train, "st")
    X_test, y_test = get_ROI_X_y_data(ROI_test, "st")
    W_train = add_vanilla_weights(ROI_train, "label_same_user_st")
    W_test = add_vanilla_weights(ROI_test, "label_same_user_st")

    lgb_train = lgb.Dataset(
        X_train, y_train,
        weight=W_train,
        feature_name=X_train.columns.to_list(),
        free_raw_data=True
    )
    lgb_test = lgb.Dataset(
        X_test, y_test,
        reference=lgb_train,
        weight=W_test,
        free_raw_data=True
    )

    with open(params_yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    print(f"starting ROI training on the segment {seg_id}...")
    eval_result = {}
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=10,
        valid_sets=lgb_train,
        callbacks=[lgb.record_evaluation(eval_result)]
    )

    # save the model:
    with open(models_saving_paths[seg_id], "wb") as fout:
        pickle.dump(gbm, fout)

    # compute metrics and save:
    y_pred_insample = gbm.predict(X_train)
    y_pred_test = gbm.predict(X_test)
    recall_insam = metric_individual_class_accuracy(y_true=y_train,
                                                    y_pred=y_pred_insample)
    recall_test = metric_individual_class_accuracy(y_true=y_test,
                                                   y_pred=y_pred_test)
    trueNegRate_insam = metric_individual_class_accuracy(y_true=y_train,
                                                    y_pred=y_pred_insample,
                                                    which_class=0)
    trueNegRate_test = metric_individual_class_accuracy(y_true=y_test,
                                                   y_pred=y_pred_test,
                                                   which_class=0)
    top5redemRate, top5profit = metric_business_top5profit(y_true=y_test,
                                                    y_pred=y_pred_test,
                                                    profit=ROI_test_bus_col)
    profit_diff_thrshld = metric_business_profit_at_diff_thrshld(
                                                    y_pred=y_pred_test,
                                                    profit=ROI_test_bus_col)
    perf = {"model for segment": seg_id,
            "eval_result": eval_result,
            "recall_in_sample": recall_insam,
            "recall_in_test": recall_test,
            "trueNegRate_in_sample": trueNegRate_insam,
            "trueNegRate_in_test": trueNegRate_test,
            "redemption rate of coupons with top raw prediction scores": top5redemRate,
            "net profit per coupon with top raw prediction scores": top5profit,
            "net profit per predicted-as-redeemed coupon at different thresholds": profit_diff_thrshld,
            "thresholds for metric computation": [0.25, 0.5, 0.75]}

    with open(perf_saving_path, "a") as f:
        f.write(str(perf))
    
    # record predictions
    pred_dict = {"y_true": y_test,
                 "Actual_pay_cent": ROI_test_bus_col,
                 "y_pred": y_pred_test}
    if seg_id == "seg_001":
        df = pd.DataFrame.from_dict(pred_dict)
    else:
        df_tmp = pd.DataFrame.from_dict(pred_dict)
        df = pd.concat([df, df_tmp], ignore_index=True)

df.to_csv(test_pred_saving_path, index=False)