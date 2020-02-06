# %%
import pdb
import pickle

import numpy as np
import pandas as pd
import pandas_profiling as pdp
from IPython.display import display
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from data import Data


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 200)


data = Data()
train_x, train_y, test_x = data.processing()

va_preds_list = []
test_preds_list = []
va_idxes_list = []

kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x, train_y))[0]
tr_x = train_x.iloc[tr_idx]
tr_y = train_y.iloc[tr_idx]
va_x = train_x.iloc[va_idx]
va_y = train_y.iloc[va_idx]

# train
feature_names = tr_x.columns

dtrain = xgb.DMatrix(tr_x, label=tr_y, feature_names=feature_names)
dvalid = xgb.DMatrix(va_x, label=va_y, feature_names=feature_names)

display(dtrain)

params = {'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'silent': 1,
          'random_state': 71}
evals = [(dtrain, 'train'), (dvalid, 'eval')]
evals_result = {}
model_xgb = xgb.train(params,
                      dtrain,
                      num_boost_round=1000,
                      early_stopping_rounds=10,
                      evals=evals,
                      evals_result=evals_result)
importance = model_xgb.get_score(importance_type='gain')
df_importance = pd.DataFrame(
    importance.values(), index=importance.keys(), columns=['importance'])
df_importance = df_importance.sort_values('importance', ascending=False)
display(df_importance)

# predict
va_pred_proba = model_xgb.predict(dvalid)
va_preds_list.append(va_pred_proba)

va_pred_binary = np.where(va_pred_proba > 0.5, 1, 0)
va_acc = accuracy_score(va_y, va_pred_binary)*100
print(f'va_acc: {round(va_acc, 2)}')

# with open(f'../output/model.pkl', 'wb') as f:
# pickle.dump(model_xgb, f)
# %%
