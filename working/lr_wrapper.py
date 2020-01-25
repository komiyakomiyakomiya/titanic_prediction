# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.pipeline import Pipeline

# from working.data import Data
# from working.cv import CV


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
        pred = self.model.predict_proba(x)[:, 1]

        """ use pipeline """
        # pred = self.pipeline.predict(x)

        return pred


# %%
if __name__ == '__main__':
    cv = CV()
    data = Data()
    lr_rgrs_wrapper = LogisticRegrWrapper()

    X_train, y_train, test, submission_temp = data.processing()

    level2_tr_pred_lr_rgrs, level2_tr_test_lr_rgrs = cv.predict_cv(
        lr_rgrs_wrapper, X_train, y_train, test)

    print(
        f'accuracy_score logistic_regression: {round(accuracy_score(y_train, level2_tr_pred_lr_rgrs)*100, 2)}')

    submission_temp['Survived'] = level2_tr_test_lr_rgrs
    submission_temp.to_csv('4fold_lr.csv', index=False)


# %%
