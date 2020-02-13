# %%
import os
import pickle

import pandas as pd
import xgboost as xgb


cwd = os.path.dirname(os.path.abspath(__file__))


class XGBWrapper(object):
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        feature_names = ['f0', 'f1']
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

        importance = self.model.get_score(importance_type='gain')
        df_importance = pd.DataFrame(
            importance.values(), index=importance.keys(), columns=['importance'])
        df_importance = df_importance.sort_values(
            'importance', ascending=False)
        print(df_importance)

    def save_model(self):
        model_dir_path = '{}/models/'.format(cwd)
        file_name_bst = 'model.bst'
        file_name_pkl = 'model.pkl'
        with open(model_dir_path+file_name_pkl, 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


if __name__ == '__main__':
    """ test """
    import numpy as np
    from data import Data
    from valid import Valid
    from xgb_wrapper import XGBWrapper
    from sklearn.metrics import accuracy_score

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth', 200)

    data = Data()
    train_x, train_y, test_x = data.processing()

    valid = Valid()
    xgb_wrap = XGBWrapper()

    pred_train_xgb, pred_test_xgb = valid.hold_out(
        xgb_wrap, train_x, train_y, test_x)

    # train acc
    pred_binary_train_xgb = np.where(pred_train_xgb > 0.5, 1, 0)
    acc_train_xgb = round(accuracy_score(
        train_y, pred_binary_train_xgb)*100, 2)
    print('##### acc_xgb #####')
    print(acc_train_xgb)

    # test pred
    pred_binary_test_xgb = np.where(pred_test_xgb > 0.5, 1, 0)
    test_x['Survived'] = pred_binary_test_xgb
    test_pred_series = test_x['Survived']
    print('##### test_pred_series #####')
    print(test_pred_series)

# %%
