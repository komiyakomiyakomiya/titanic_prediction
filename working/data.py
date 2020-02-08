# %%
import os

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# %%
class Data(object):
    def __init__(self):
        self.CWD = os.path.dirname(os.path.abspath(__file__))
        print(self.CWD)

    def _load(self):
        train_path = f'{self.CWD}/../input/titanic/train.csv'
        train = pd.read_csv(train_path, error_bad_lines=False)
        test_path = f'{self.CWD}/../input/titanic/test.csv'
        test = pd.read_csv(test_path, error_bad_lines=False)
        return train, test

    def processing(self):
        train, test = self._load()
        le = LabelEncoder()
        le.fit(train['Sex'])

        train['Sex'] = le.transform(train['Sex'])
        train_dropna = train.dropna()
        train_x = train_dropna[['Age', 'Sex']]
        train_y = train_dropna['Survived']

        test['Sex'] = le.transform(test['Sex'])
        test_dropna = test.dropna()
        test_x = test_dropna[['Age', 'Sex']]

        return train_x, train_y, test_x

    def nn_processing(self):
        train, test = self._load()

        train_dropna = train.dropna()
        train_x = train_dropna[['Age', 'Sex']]
        train_x = pd.get_dummies(train_x)
        train_y = train_dropna['Survived']

        test_dropna = test.dropna()
        test_x = test_dropna[['Age', 'Sex']]
        test_x = pd.get_dummies(test_x)

        return train_x, train_y, test_x


if __name__ == '__main__':
    """ test """
    import sys

    from IPython.display import display

    data = Data()
    train_x, train_y, test_x = data.processing()
    print('################################')
    print(sys.path)
