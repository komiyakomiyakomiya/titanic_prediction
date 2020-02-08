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
        print(df_importance)

        # [print(i) for i in sorted(feature_importance.items(), key=lambda x:)]

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred