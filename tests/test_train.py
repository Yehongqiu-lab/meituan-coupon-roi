from src.train import _edit_filter, load_policy_certain_segment_data, metric_individual_class_accuracy
import os
import sys
import pandas as pd
import numpy as np
import pytest

def test_filter_editor():
    """
    test multiple cases:
    1. input filter is [[]], should return [[arg]]
    2. input filter is [[a1], [a2]], should return [[a1, arg], [a2, arg]]
    """
    new_filter_1 = _edit_filter([[]], ("test_tuple", "==", 1))
    new_filter_2 = _edit_filter([[("a1", ">", 1), ("a1", "==", 2)], [("a2", ">", 3)]], ("test_tuple", "==", 1))

    assert new_filter_1 == [[("test_tuple", "==", 1)]]
    assert new_filter_2 == [[("a1", ">", 1), ("a1", "==", 2), ("test_tuple", "==", 1)], 
                            [("a2", ">", 3), ("test_tuple", "==", 1)]]

def test_loading_segmented_data():
    repo_root = os.path.dirname(os.getcwd())

    segmented_data_1 = load_policy_certain_segment_data(repo_root,
                                                        train_set_path="data_work/policy_train_set.parquet")
    subset_1 = segmented_data_1.loc[(segmented_data_1["Price_limit_cent"] > 10000)
                                    & (segmented_data_1["Coupon_amt_cent"] <= 1000)]
    
    segmented_data_2 = load_policy_certain_segment_data(repo_root,
                                                        Price_limit_bin=[1, 2],
                                                        Coupon_limit_bin=[0, 1],
                                                        Expiry_span_bin=[0, 0],
                                                        train_set_path="data_work/policy_train_set.parquet")
    subset_2 = segmented_data_2.loc[(segmented_data_2["Price_limit_cent"] > 1000)
                                    & (segmented_data_2["Coupon_amt_cent"] <= 10000)]
    
    assert segmented_data_1.shape[0] == subset_1.shape[0]
    assert segmented_data_2.shape[0] == subset_2.shape[0]

def test_metric_individual_class_accuracy():
    output_1 = metric_individual_class_accuracy(pd.Series([0, 0, 0, 1, 1, 1]), np.array([0.15, 0.45, 0.7, 0.3, 0.6, 0.9]), test_mode=True)
    output_2 = metric_individual_class_accuracy([0, 0, 0, 1, 1, 1], [0.15, 0.45, 0.7, 0.3, 0.6, 0.9], which_class=0, test_mode=True)
    assert output_1 == pytest.approx([1, 2/3, 1/3])
    assert output_2 == pytest.approx([1/3, 2/3, 1])