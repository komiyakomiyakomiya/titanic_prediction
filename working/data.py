# %%
import os
from io import BytesIO

from google.cloud import storage
import numpy as np
import pandas as pd

if '/Users/' in __file__:
    # from dotenv import load_dotenv
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    # load_dotenv('{}/../.env'.format(cwd))


class Data(object):
    def __init__(self):
        pass

    def load(self):
        TRAIN_PATH = '{}/../input/titanic/train.csv'.format(cwd)
        TEST_PATH = '{}/../input/titanic/test.csv'.format(cwd)

        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        return train, test

    def load_gcs(self):
        PROJECT_NAME = 'aipsample'
        BUCKET_NAME = 'aipsamplebucket'
        TRAIN = 'input/train.csv'
        TEST = 'input/test.csv'
        client = storage.Client(PROJECT_NAME)
        bucket = client.get_bucket(BUCKET_NAME)
        blob_train = storage.Blob(TRAIN, bucket)
        blob_test = storage.Blob(TEST, bucket)
        data_train = blob_train.download_as_string()
        data_test = blob_test.download_as_string()
        train = pd.read_csv(BytesIO(data_train))
        test = pd.read_csv(BytesIO(data_test))
        return train, test

    def processing(self):
        train, test = self.load()
        sex_dict = {'male': 0, 'female': 1}

        train['f0'] = train['Age']
        train['f1'] = train['Sex'].map(sex_dict)
        train_dropna = train.dropna()
        train_x = train_dropna[['f0', 'f1']]
        train_y = train_dropna['Survived']

        test['f0'] = test['Age']
        test['f1'] = test['Sex'].map(sex_dict)
        test_dropna = test.dropna()
        test_x = test_dropna[['f0', 'f1']]

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
