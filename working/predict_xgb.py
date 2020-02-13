# %%
import os
import sys
import pdb

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

self_dir_path = os.path.dirname(os.path.abspath(__file__))
sys_path_list = sys.path
if self_dir_path not in sys_path_list:
    sys_path_list.append(self_dir_path)

if True:
    from data import Data
    from valid import Valid
    from xgb_wrapper import XGBWrapper


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', 200)


data = Data()
train_x, train_y, test_x = data.processing()

valid = Valid()
xgb_wrap = XGBWrapper()

pred_train_xgb, pred_test_xgb = valid.hold_out(
    xgb_wrap, train_x, train_y, test_x)

# train acc
pred_binary_train_xgb = np.where(pred_train_xgb > 0.5, 1, 0)
acc_train_xgb = round(accuracy_score(train_y, pred_binary_train_xgb)*100, 2)
print('##### acc_xgb #####')
print(acc_train_xgb)

# test pred
# pred_binary_test_xgb = np.where(pred_test_xgb > 0.5, 1, 0)
# test_x['Survived'] = pred_binary_test_xgb
# test_pred_series = test_x['Survived']
# print('##### test_pred_series #####')
# print(test_pred_series)
