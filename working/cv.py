import numpy as np
from sklearn.model_selection import StratifiedKFold
import pdb


class CV(object):
    def __init__(self):
        pass

    def predict_cv(self, model_wrapper, train_x, train_y, test_x):
        preds_list_valid = []
        preds_test_list = []
        index_list_valid = []

        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
        for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y), 1):
            print(f'fold{i} start')
            tr_x = train_x.iloc[tr_idx]
            tr_y = train_y.iloc[tr_idx]
            va_x = train_x.iloc[va_idx]
            va_y = train_y.iloc[va_idx]

            model_wrapper.fit(tr_x, tr_y, va_x, va_y)

            pred_valid = model_wrapper.predict(va_x)
            preds_list_valid.append(pred_valid)
            pred_test = model_wrapper.predict(test_x)
            preds_test_list.append(pred_test)
            index_list_valid.append(va_idx)
            print(f'fold{i} end\n')

        index_list_valid = np.concatenate(index_list_valid, axis=0)
        preds_list_valid = np.concatenate(preds_list_valid, axis=0)
        order = np.argsort(index_list_valid)
        pred_train = preds_list_valid[order]
        pred_test_mean = np.mean(preds_test_list, axis=0)

        # trは特徴量にするのでmeanする必要ない
        return pred_train, pred_test_mean
