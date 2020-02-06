# %%
import pickle

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt


class XGBWrapper(object):
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        feature_names = tr_x.columns
        dtrain = xgb.DMatrix(tr_x, label=tr_y, feature_names=feature_names)
        dvalid = xgb.DMatrix(va_x, label=va_y, feature_names=feature_names)

        params = {'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'silent': 1,
                  'random_state': 71}

        evals = [(dtrain, 'train'), (dvalid, 'eval')]
        evals_result = {}

        self.model = xgb.train(params,
                               dtrain,
                               num_boost_round=100,
                               early_stopping_rounds=10,
                               evals=evals,
                               evals_result=evals_result)

        # ラウンド毎の損失の減少を可視化
        # train_metric = evals_result['train']['logloss']
        # plt.plot(train_metric, label='train logloss')
        # eval_metric = evals_result['eval']['logloss']
        # plt.plot(eval_metric, label='eval logloss')
        # plt.grid()
        # plt.legend()
        # plt.xlabel('rounds')
        # plt.ylabel('logloss')
        # plt.show()

        importance = self.model.get_score(importance_type='gain')
        df_importance = pd.DataFrame(
            importance.values(), index=importance.keys(), columns=['importance'])
        # 降順にソート
        df_importance = df_importance.sort_values(
            'importance', ascending=False)
        display(df_importance)

        # [print(i) for i in sorted(feature_importance.items(), key=lambda x:)]

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import accuracy_score

    from data import Data
    from cv import CV

    data = Data()
    cv = CV()
    xgb_wrap = XGBWrapper()

    X_train, y_train, X_test = data.processing()

    tr_pred, test_pred = cv.predict_cv(
        xgb_wrap, X_train, y_train, X_test)

    tr_pred_binary = np.where(tr_pred > 0.5, 1, 0)
    test_pred_binary = np.where(test_pred > 0.5, 1, 0)

    print(accuracy_score(y_train, tr_pred_binary)*100, 2)
    print(accuracy_score(y_test, test_pred_binary)*100, 2)


# %%
