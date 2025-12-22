"""
import sys
import os
import importlib

root = os.path.dirname(os.getcwd())
repo_root = os.path.join(root, "meituan-coupon-roi")
src_path = os.path.join(repo_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

from combine_features import *
# combine features and output to the trainable.parquet
features_combining(
    os.path.join(repo_root, "data_work/rcs_w_cpn_features.parquet"),
    os.path.join(repo_root, "data_work/rcs_w_user_features.parquet"),
    os.path.join(repo_root, "data_work/trainable.parquet")
)
"""

import sys
import os
import importlib

root = os.path.dirname(os.getcwd())
repo_root = os.path.join(root, "meituan-coupon-roi")
src_path = os.path.join(repo_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
from splitting import *

# data splitting
"""
policy_model_split(
    os.path.join(repo_root, "data_work/trainable.parquet"),
    os.path.join(repo_root, "data_work/policy_train_set_w_CV.parquet"),
    os.path.join(repo_root, "data_work/policy_test_set.parquet")
)
"""

ROI_model_split(
    os.path.join(repo_root, "data_work/trainable.parquet"),
    os.path.join(repo_root, "data_work/ROI_train_set_w_CV.parquet"),
    os.path.join(repo_root, "data_work/ROI_test_set.parquet")
)
