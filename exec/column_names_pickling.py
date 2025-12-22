import pickle
from pathlib import Path

import os
import sys
root = os.path.dirname(os.getcwd())
repo_root = os.path.join(root, "meituan-coupon-roi")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from io_load import *
df = load_df_from_pq("meituan-coupon-roi/data_work/policy_train_set_w_CV.parquet", 
                     filters=[("receipt_key", "==", 1)])
cols = df.columns.to_list()

pickle_path = os.path.join(repo_root, "data_work/trainable_colnames.pkl")
with open(pickle_path, 'wb') as f:
    pickle.dump(cols, f)
