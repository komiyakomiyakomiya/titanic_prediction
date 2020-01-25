# %%
from catboost import CatBoost
from catboost import Pool


class CatWrapper(object):
    def __init__(self):
        self.model = None
        self.params = {
            'loss_function': 'Logloss',
            'num_boost_round': 100
        }

    def fit(self, tr_x, tr_y, va_x, va_y):
        train_pool = Pool(tr_x, tr_y)
        valid_pool = Pool(va_x, va_y)

        self.model = CatBoost(self.params)
        self.model.fit(train_pool)

    def predict(self, x):
        data = Pool(x)
        pred = self.model.predict(
            data, prediction_type='RawFormulaVal')
        # prediction_type -> 'Class', 'Probability', 'RawFormulaVal'
        return pred
