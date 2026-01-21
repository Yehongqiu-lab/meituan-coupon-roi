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

from train import *
id_param = "1217"

date = "0106"
m_id = "m_7d"

seg_params = []
for p in range(3):
    for c in range(3):
        for e in range(2):
            if [p, c, e] not in [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 2, 1], [0, 0, 0]]:
                seg_params.append([p, c, e])

seg_id = "oth_segs"

params_yaml_path = os.path.join(repo_root, f"conf/params_ROI_{id_param}.yaml")

model_saving_path_1 = os.path.join(repo_root, f"models/ROI/ROI_{date}_{m_id}_{seg_id}_fd1.pkl")
perf_saving_path_1 = os.path.join(repo_root, f"experiment_results/ROI/perf_ROI_{date}_{m_id}_{seg_id}_fd1.txt")
val_pred_saving_path_1 = os.path.join(repo_root, f"experiment_results/ROI/val_pred_ROI_{date}_{m_id}_{seg_id}_fd1.csv")

model_saving_path_2 = os.path.join(repo_root, f"models/ROI/ROI_{date}_{m_id}_{seg_id}_fd2.pkl")
perf_saving_path_2 = os.path.join(repo_root, f"experiment_results/ROI/perf_ROI_{date}_{m_id}_{seg_id}_fd2.txt")
val_pred_saving_path_2 = os.path.join(repo_root, f"experiment_results/ROI/val_pred_ROI_{date}_{m_id}_{seg_id}_fd2.csv")

model_saving_path_3 = os.path.join(repo_root, f"models/ROI/ROI_{date}_{m_id}_{seg_id}_fd3.pkl")
perf_saving_path_3 = os.path.join(repo_root, f"experiment_results/ROI/perf_ROI_{date}_{m_id}_{seg_id}_fd3.txt")
val_pred_saving_path_3 = os.path.join(repo_root, f"experiment_results/ROI/val_pred_ROI_{date}_{m_id}_{seg_id}_fd3.csv")

print("Loading data for ROI training...")
ROI_train_1, ROI_train_2, ROI_train_3, ROI_val_1, ROI_val_2, ROI_val_3 = \
    load_ROI_training_3folds_data(repo_root,
                                     pickle_path=f"conf/feature_selection/ROI_modeling/ROI_train_cols_{m_id}.pkl",
                                     segment=True,
                                     Price_limit_bin=seg_params[0][0],
                                     Coupon_limit_bin=seg_params[0][1],
                                     Expiry_span_bin=seg_params[0][2])

for seg_param in seg_params[1:]:
    train_1, train_2, train_3, val_1, val_2, val_3 = \
    load_ROI_training_3folds_data(repo_root,
                                     pickle_path=f"conf/feature_selection/ROI_modeling/ROI_train_cols_{m_id}.pkl",
                                     segment=True,
                                     Price_limit_bin=seg_param[0],
                                     Coupon_limit_bin=seg_param[1],
                                     Expiry_span_bin=seg_param[2])
    ROI_train_1 = pd.concat([ROI_train_1, train_1], ignore_index=True)
    ROI_train_2 = pd.concat([ROI_train_2, train_2], ignore_index=True)
    ROI_train_3 = pd.concat([ROI_train_3, train_3], ignore_index=True)
    ROI_val_1 = pd.concat([ROI_val_1, val_1], ignore_index=True)
    ROI_val_2 = pd.concat([ROI_val_2, val_2], ignore_index=True)
    ROI_val_3 = pd.concat([ROI_val_3, val_3], ignore_index=True)

X_train_1, y_train_1 = get_ROI_X_y_data(ROI_train_1, "st")
X_train_2, y_train_2 = get_ROI_X_y_data(ROI_train_2, "st")
X_train_3, y_train_3 = get_ROI_X_y_data(ROI_train_3, "st")

W_train_1 = add_vanilla_weights(ROI_train_1, "label_same_user_st")
W_train_2 = add_vanilla_weights(ROI_train_2, "label_same_user_st")
W_train_3 = add_vanilla_weights(ROI_train_3, "label_same_user_st")

X_val_1, y_val_1 = get_ROI_X_y_data(ROI_val_1, "st")
X_val_2, y_val_2 = get_ROI_X_y_data(ROI_val_2, "st")
X_val_3, y_val_3 = get_ROI_X_y_data(ROI_val_3, "st")

W_val_1 = add_vanilla_weights(ROI_val_1, "label_same_user_st")
W_val_2 = add_vanilla_weights(ROI_val_2, "label_same_user_st")
W_val_3 = add_vanilla_weights(ROI_val_3, "label_same_user_st")


lgb_train_1 = lgb.Dataset(
    X_train_1, y_train_1, 
    weight=W_train_1,
    feature_name=X_train_1.columns.to_list(),
    free_raw_data=True
)
lgb_val_1 = lgb.Dataset(
    X_val_1, y_val_1, reference=lgb_train_1, weight=W_val_1, free_raw_data=True
)

lgb_train_2 = lgb.Dataset(
    X_train_2, y_train_2,
    weight=W_train_2, 
    feature_name=X_train_2.columns.to_list(),
    free_raw_data=True
)
lgb_val_2 = lgb.Dataset(
    X_val_2, y_val_2, reference=lgb_train_2, weight=W_val_2, free_raw_data=True
)

lgb_train_3 = lgb.Dataset(
    X_train_3, y_train_3, 
    weight=W_train_3,
    feature_name=X_train_3.columns.to_list(),
    free_raw_data=True
)
lgb_val_3 = lgb.Dataset(
    X_val_3, y_val_3, reference=lgb_train_3, weight=W_val_3, free_raw_data=True
)

with open(params_yaml_path, 'r') as f:
    params = yaml.safe_load(f)

print("starting ROI training with cross validation...")

eval_result_1 = {}
gbm_1 = lgb.train(
    params,
    lgb_train_1,
    num_boost_round=10,
    valid_sets=lgb_val_1,
    callbacks=[lgb.record_evaluation(eval_result_1)]
)

"""gbm_1.save_model(model_saving_path_1, num_iteration=0)"""

with open(model_saving_path_1, "wb") as fout:
    pickle.dump(gbm_1, fout)

recall_in_sample_1 = metric_individual_class_accuracy(y_true=y_train_1,
                                                      y_pred=gbm_1.predict(X_train_1))
recall_val_1 = metric_individual_class_accuracy(y_true=y_val_1,
                                                y_pred=gbm_1.predict(X_val_1))
trueNegRate_in_sample_1 = metric_individual_class_accuracy(y_true=y_train_1,
                                                      y_pred=gbm_1.predict(X_train_1),
                                                      which_class=0)
trueNegRate_val_1 = metric_individual_class_accuracy(y_true=y_val_1,
                                                      y_pred=gbm_1.predict(X_val_1),
                                                      which_class=0)

pred_dict = {"y_true": y_val_1, "y_pred": gbm_1.predict(X_val_1)}
df = pd.DataFrame.from_dict(pred_dict)
df.to_csv(val_pred_saving_path_1, index=False)

perf_1 = {"eval_result": eval_result_1,
          "recall_in_sample": recall_in_sample_1,
          "recall_val": recall_val_1,
          "trueNegRate_in_sample": trueNegRate_in_sample_1,
          "trueNegRate_val": trueNegRate_val_1,
          "thresholds for metric computation": [0.25, 0.5, 0.75]}
with open(perf_saving_path_1, "w") as f:
    f.write(str(perf_1))

eval_result_2 = {}
gbm_2 = lgb.train(
    params,
    lgb_train_2,
    num_boost_round=10,
    valid_sets=lgb_val_2,
    callbacks=[lgb.record_evaluation(eval_result_2)]
)
"""gbm_2.save_model(model_saving_path_2, num_iteration=0)"""

with open(model_saving_path_2, "wb") as fout:
    pickle.dump(gbm_2, fout)

recall_in_sample_2 = metric_individual_class_accuracy(y_true=y_train_2,
                                                      y_pred=gbm_2.predict(X_train_2))
recall_val_2 = metric_individual_class_accuracy(y_true=y_val_2,
                                                y_pred=gbm_2.predict(X_val_2))

trueNegRate_in_sample_2 = metric_individual_class_accuracy(y_true=y_train_2,
                                                      y_pred=gbm_2.predict(X_train_2),
                                                      which_class=0)
trueNegRate_val_2 = metric_individual_class_accuracy(y_true=y_val_2,
                                                      y_pred=gbm_2.predict(X_val_2),
                                                      which_class=0)

pred_dict = {"y_true": y_val_2, "y_pred": gbm_1.predict(X_val_2)}
df = pd.DataFrame.from_dict(pred_dict)
df.to_csv(val_pred_saving_path_2, index=False)

perf_2 = {"eval_result": eval_result_2,
          "recall_in_sample": recall_in_sample_2,
          "recall_val": recall_val_2,
          "trueNegRate_in_sample": trueNegRate_in_sample_2,
          "trueNegRate_val": trueNegRate_val_2,
          "thresholds for metric computation": [0.25, 0.5, 0.75]}
with open(perf_saving_path_2, "w") as f:
    f.write(str(perf_2))


eval_result_3 = {}
gbm_3 = lgb.train(
    params,
    lgb_train_3,
    num_boost_round=10,
    valid_sets=lgb_val_3,
    callbacks=[lgb.record_evaluation(eval_result_3)]
)
"""gbm_3.save_model(model_saving_path_3, num_iteration=0)"""

with open(model_saving_path_3, "wb") as fout:
    pickle.dump(gbm_3, fout)

recall_in_sample_3 = metric_individual_class_accuracy(y_true=y_train_3,
                                                      y_pred=gbm_3.predict(X_train_3))
recall_val_3 = metric_individual_class_accuracy(y_true=y_val_3,
                                                y_pred=gbm_3.predict(X_val_3))

trueNegRate_in_sample_3 = metric_individual_class_accuracy(y_true=y_train_3,
                                                      y_pred=gbm_3.predict(X_train_3),
                                                      which_class=0)
trueNegRate_val_3 = metric_individual_class_accuracy(y_true=y_val_3,
                                                      y_pred=gbm_3.predict(X_val_3),
                                                      which_class=0)

pred_dict = {"y_true": y_val_3, "y_pred": gbm_1.predict(X_val_3)}
df = pd.DataFrame.from_dict(pred_dict)
df.to_csv(val_pred_saving_path_3, index=False)

perf_3 = {"eval_result": eval_result_3,
          "recall_in_sample": recall_in_sample_3,
          "recall_val": recall_val_3,
          "trueNegRate_in_sample": trueNegRate_in_sample_3,
          "trueNegRate_val": trueNegRate_val_3,
          "thresholds for metric computation": [0.25, 0.5, 0.75]}
with open(perf_saving_path_3, "w") as f:
    f.write(str(perf_3))



"""
print("start ROI training based on all available training data...")
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