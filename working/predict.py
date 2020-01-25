# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
# from sklearn.metrics import log_loss

try:
    from working.data import Data
    from working.cv import CV
    from working.lgbm_wrapper import LGBMWrapper
    from working.xgb_wrapper import XGBWrapper
    from working.cat_wrapper import CatWrapper
    from working.lr_wrapper import LogisticRegrWrapper
except ImportError:
    from data import Data
    from cv import CV
    from lgbm_wrapper import LGBMWrapper
    from xgb_wrapper import XGBWrapper
    from cat_wrapper import CatWrapper
    from lr_wrapper import LogisticRegrWrapper


data = Data()
X_train, y_train, test, submission_temp = data.processing()

cv = CV()

lgbm_wrap = LGBMWrapper()
level1_tr_pred_lgbm, level1_test_pred_lgbm = cv.predict_cv(
    lgbm_wrap, X_train, y_train, test)

xgb_wrap = XGBWrapper()
level1_tr_pred_xgb, level1_test_pred_xgb = cv.predict_cv(
    xgb_wrap, X_train, y_train, test)

cat_wrap = CatWrapper()
level1_tr_pred_cat, level1_test_pred_cat = cv.predict_cv(
    cat_wrap, X_train, y_train, test)

print(
    f'accuracy_score lightgbm: {round(accuracy_score(y_train, (level1_tr_pred_lgbm > 0.5).astype(int))*100, 2)}')

print(
    f'accuracy_score xgboost: {round(accuracy_score(y_train, (level1_tr_pred_xgb > 0.5).astype(int))*100, 2)}')

print(
    f'accuracy_score catboost: {round(accuracy_score(y_train, (level1_tr_pred_cat > 0.5).astype(int))*100, 2)}')


X_train2 = pd.DataFrame({'pred_lgbm': level1_tr_pred_lgbm,
                         'pred_xgb': level1_tr_pred_xgb,
                         'pred_cat': level1_tr_pred_cat})

test2 = pd.DataFrame({'pred_lgbm': level1_test_pred_lgbm,
                      'pred_xgb': level1_test_pred_xgb,
                      'pred_cat': level1_test_pred_cat})

lr_wrap = LogisticRegrWrapper()
level2_tr_pred_lr, level2_test_pred_lr = cv.predict_cv(
    lr_wrap, X_train2, y_train, test2)

print(
    f'accuracy_score LogisticRegrssion: {round(accuracy_score(y_train, (level2_tr_pred_lr > 0.5).astype(int))*100, 2)}')


submission_temp['Survived'] = (level2_test_pred_lr > 0.5).astype(int)
submission_temp.to_csv('submission.csv', index=False)
