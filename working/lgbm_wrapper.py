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
        print(importance)

    def predict(self, x):
        pred_proba = self.model.predict(x)
        # pred_class = (pred > 0.5).astype(int)
        return pred_proba