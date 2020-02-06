# %%
import pdb

import numpy as np
import pandas as pd
import pandas_profiling as pdp
from IPython.display import display
from sklearn.metrics import accuracy_score

from data import Data
from cv import CV
from lgbm_wrapper import LGBMWrapper
from xgb_wrapper import XGBWrapper
from cat_wrapper import CatWrapper
from lr_wrapper import LogisticRegrWrapper
from nn_wrapper import NNWrapper
import os


print('#######predict#######')
print(os.getcwd())
print('#######predict#######')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 200)

data = Data()
train_x, train_y, test_x = data.processing()

cv = CV()


# stacking layer1
# LightGBM
print('######### LGBM #########')
lgbm_wrap = LGBMWrapper()
pred_train_lgbm, pred_test_lgbm = cv.predict_cv(
    lgbm_wrap, train_x, train_y, test_x)

pred_binary_train_lgbm = np.where(pred_train_lgbm > 0.5, 1, 0)
pred_binary_test_lgbm = np.where(pred_test_lgbm > 0.5, 1, 0)

acc_train_lgbm = round(accuracy_score(train_y, pred_binary_train_lgbm)*100, 2)

print(f'acc_train LGBM: {acc_train_lgbm}')


# XGBoost
print('######### XGB #########')
xgb_wrap = XGBWrapper()
pred_train_xgb, pred_test_xgb = cv.predict_cv(
    xgb_wrap, train_x, train_y, test_x)

pred_binary_train_xgb = np.where(pred_train_xgb > 0.5, 1, 0)
pred_binary_test_xgb = np.where(pred_test_xgb > 0.5, 1, 0)

acc_train_xgb = round(accuracy_score(train_y, pred_binary_train_xgb)*100, 2)

print(f'acc_train XGB: {acc_train_xgb}')

# %%
# CatBoost
print('######### Cat #########')
cat_wrap = CatWrapper()
pred_train_cat, pred_test_cat = cv.predict_cv(
    cat_wrap, train_x, train_y, test_x)

pred_binary_train_cat = np.where(pred_train_cat > 0.5, 1, 0)
pred_binary_test_cat = np.where(pred_test_cat > 0.5, 1, 0)

acc_train_cat = round(accuracy_score(train_y, pred_binary_train_cat)*100, 2)

print(f'acc_train Cat: {acc_train_cat}')


# NeuralNetwork
print('######### NN #########')
nn_wrap = NNWrapper()
pred_train_nn, pred_test_nn = cv.predict_cv(
    nn_wrap, train_x, train_y, test_x)

pred_binary_train_nn = np.where(pred_train_nn > 0.5, 1, 0)
pred_binary_test_nn = np.where(pred_test_nn > 0.5, 1, 0)

acc_train_nn = round(accuracy_score(train_y, pred_binary_train_nn)*100, 2)

print(f'acc_train NN: {acc_train_nn}')


# Make feature for stacking layer2
train_x2 = pd.DataFrame({'pred_lgbm': pred_train_lgbm,
                         'pred_xgb': pred_train_xgb,
                         'pred_cat': pred_train_cat,
                         'pred_nn': pred_train_nn})

test_x2 = pd.DataFrame({'pred_lgbm': pred_test_lgbm,
                        'pred_xgb': pred_test_xgb,
                        'pred_cat': pred_test_cat,
                        'pred_nn': pred_test_nn})


# stacking layer2
# LogisticRegrssor
print('######### LR #########')
lr_wrap = LogisticRegrWrapper()
pred_train_lr, pred_test_lr = cv.predict_cv(
    lr_wrap, train_x2, train_y, test_x2)

pred_binary_train_lr = np.where(pred_train_lr > 0.5, 1, 0)
pred_binary_test_lr = np.where(pred_test_lr > 0.5, 1, 0)

acc_train_lr = round(accuracy_score(train_y, pred_binary_train_lr)*100, 2)

print(f'acc_train LR: {acc_train_lr}')

# %%
