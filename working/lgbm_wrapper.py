# %%
import lightgbm as lgb


class LGBMWrapper(object):
    def __init__(self):
        self.model = None
        self.params = {'objective': 'binary',
                       'seed': 71,
                       'verbose': 0}

    def fit(self, tr_x, tr_y, va_x, va_y):
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)
        self.model = lgb.train(params=self.params,
                               train_set=lgb_train,
                               valid_sets=[lgb_train, lgb_valid],
                               early_stopping_rounds=20,
                               verbose_eval=-1)

    def predict(self, x):
        pred = self.model.predict(x)
        # pred_class = (pred > 0.5).astype(int)
        return pred
