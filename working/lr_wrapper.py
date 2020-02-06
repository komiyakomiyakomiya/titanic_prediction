# %%
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline


class LogisticRegrWrapper(object):
    def __init__(self):
        self.model = None
        self.scaler = None
        # self.pipeline = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        tr_x = tr_x.fillna(tr_x.mean())

        """ not use pipeline """
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(solver='lbfgs', C=1.0, random_state=71)
        self.model.fit(tr_x, tr_y)

        """ use pipeline """
        # self.pipeline = Pipeline([
        #     ('scl', StandardScaler()),
        #     ('est', LogisticRegression(solver='lbfgs', C=1.0, random_state=71))
        # ])
        # self.pipeline.fit(tr_x, tr_y)

    def predict(self, x):
        x = x.fillna(x.mean())

        """ not use pipeline """
        x = self.scaler.transform(x)
        pred_proba = self.model.predict_proba(x)[:, 1]

        """ use pipeline """
        # pred = self.pipeline.predict(x)

        return pred_proba


# %%
if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import accuracy_score

    from data import Data
    from cv import CV

    data = Data()
    cv = CV()
    lr_wrap = LogisticRegrWrapper()

    X_train, y_train, X_test, y_test = data.processing()

    lr_pred_tr_layer, lr_pred_test_layer = cv.predict_cv(
        lr_wrap, X_train, y_train, X_test)

    tr_pred_binary = np.where(lr_pred_tr_layer > 0.5, 1, 0)
    test_pred_binary = np.where(lr_pred_test_layer > 0.5, 1, 0)

    print(
        f'accuracy_score LogisticRegrssion: {round(accuracy_score(y_train, tr_pred_binary)*100, 2)}')

    print(
        f'accuracy_score LogisticRegrssion: {round(accuracy_score(y_test, test_pred_binary)*100, 2)}')
# %%
