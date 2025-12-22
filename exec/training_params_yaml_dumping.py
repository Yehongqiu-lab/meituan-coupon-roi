import yaml
import os
import sys
root = os.path.dirname(os.getcwd())
repo_root = os.path.join(root, "meituan-coupon-roi")

params = {
     "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0
}

yaml_path = os.path.join(repo_root, "conf/params_policy_1217.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(params, f)