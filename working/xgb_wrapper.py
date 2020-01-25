
# %%
import xgboost as xgb


class XGBWrapper(object):
    def __init__(self):
        self.model = None
        self.params = {'objective': 'binary:logistic',
                       'eval_metric': 'error',
                       'silent': 1,
                       'random_state': 71}

        self.num_round = 50

    def fit(self, tr_x, tr_y, va_x, va_y):
        dtrain = xgb.DMatrix(tr_x, tr_y)
        dvalid = xgb.DMatrix(va_x, va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        self.model = xgb.train(self.params,
                               dtrain,
                               self.num_round,
                               evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        # pred_class = (pred > 0.5).astype(int)
        return pred
