# %%
import pickle

import pandas as pd
from catboost import CatBoost
from catboost import Pool


class CatWrapper(object):
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        train_pool = Pool(tr_x, tr_y)
        valid_pool = Pool(va_x, va_y)

        params = {
            'loss_function': 'Logloss',
            'num_boost_round': 1000,
            'early_stopping_rounds': 10,
        }
        self.model = CatBoost(params)
        self.model.fit(train_pool)

        importance = pd.DataFrame(self.model.get_feature_importance(
        ), index=tr_x.columns, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        print(importance)

    def predict(self, x):
        data = Pool(x)
        pred_proba = self.model.predict(
            data, prediction_type='RawFormulaVal')
        # prediction_type -> 'Class', 'Probability', 'RawFormulaVal'
        return pred_proba