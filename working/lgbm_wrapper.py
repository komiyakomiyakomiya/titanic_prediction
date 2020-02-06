# %%
import pickle

import lightgbm as lgb
import pandas as pd


class LGBMWrapper(object):
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)

        params = {'objective': 'binary',
                  'seed': 71,
                  'verbose': 0}
        self.model = lgb.train(params=params,
                               train_set=lgb_train,
                               valid_sets=[lgb_train, lgb_valid],
                               early_stopping_rounds=10,
                               verbose_eval=-1)

        # importanceを表示する
        importance = pd.DataFrame(self.model.feature_importance(
        ), index=tr_x.columns, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        display(importance)

    def predict(self, x):
        pred_proba = self.model.predict(x)
        # pred_class = (pred > 0.5).astype(int)
        return pred_proba


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import accuracy_score

    from data import Data
    from cv import CV

    data = Data()
    cv = CV()
    lgbm_wrap = LGBMWrapper()

    X_train, y_train, X_test, y_test = data.processing()

    tr_pred_proba, test_pred_proba = cv.predict_cv(
        lgbm_wrap, X_train, y_train, X_test)

    tr_pred_binary = np.where(tr_pred_proba > 0.5, 1, 0)
    test_pred_binary = np.where(test_pred_proba > 0.5, 1, 0)

    print(accuracy_score(y_train, tr_pred_binary)*100, 2)
    print(accuracy_score(y_test, test_pred_binary)*100, 2)


# %%
