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
print("Loading data for ROI training...")
ROI_train_1, ROI_train_2, ROI_train_3, ROI_val_1, ROI_val_2, ROI_val_3 = \
    load_ROI_training_3folds_data(repo_root, fh_or_st="st")

ROI_test = load_ROI_test_data(repo_root, fh_or_st="st")


