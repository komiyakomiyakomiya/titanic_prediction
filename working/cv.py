import numpy as np
from sklearn.model_selection import StratifiedKFold


class CV(object):
    def __init__(self):
        pass

    def predict_cv(self, model_wrapper, X_train, y_train, test):
        va_preds_list = []
        test_preds_list = []
        va_idxes_list = []

        # ######
        # va_score_list = []
        # ######

        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
        for fold_num, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train), 1):
            print(f'fold{fold_num} start')
            tr_x = X_train.iloc[tr_idx]
            tr_y = y_train[tr_idx]
            va_x = X_train.iloc[va_idx]
            va_y = y_train[va_idx]

            model_wrapper.fit(tr_x, tr_y, va_x, va_y)

            va_pred = model_wrapper.predict(va_x)
            va_preds_list.append(va_pred)
            test_pred = model_wrapper.predict(test)
            test_preds_list.append(test_pred)
            va_idxes_list.append(va_idx)
            print(f'fold{fold_num} end\n')

            # ######
            # va_pred_bool = (va_pred > 0.5).astype(int)
            # print(va_pred_bool)
            # va_score = (round(accuracy_score(va_y, va_pred_bool)*100, 2))
            # print(va_score)
            # va_score_list.append(va_score)
            # ######

        va_idxes_list = np.concatenate(va_idxes_list, axis=0)
        va_preds_list = np.concatenate(va_preds_list, axis=0)
        order = np.argsort(va_idxes_list)
        tr_pred = va_preds_list[order]
        # tr_pred_class = (tr_pred > 0.5).astype(int)
        test_pred_mean = np.mean(test_preds_list, axis=0)
        # test_pred_class = (test_pred_mean > 0.5).astype(int)

        # ######
        # va_score_mean = np.mean(va_score_list, axis=0)
        # print(va_score_list, round(va_score_mean, 2))
        # ######

        return tr_pred, test_pred
        # return tr_pred_class, test_pred_class
