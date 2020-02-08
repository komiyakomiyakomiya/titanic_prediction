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