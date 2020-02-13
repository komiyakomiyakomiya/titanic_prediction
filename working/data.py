# %%
import os
from io import BytesIO

from google.cloud import storage
import numpy as np
import pandas as pd

if '/Users/' in __file__:
    from dotenv import load_dotenv
    CWD = os.path.dirname(os.path.abspath(__file__))
    print(CWD)
    load_dotenv('{}/../.env'.format(CWD))


class Data(object):
    def __init__(self):
        pass

    def load(self):
        TRAIN_PATH = '{}/../input/titanic/train.csv'.format(CWD)
        TEST_PATH = '{}/../input/titanic/test.csv'.format(CWD)

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


if __name__ == '__main__':
    """ test """
    import sys

    data = Data()
    train_x, train_y, test_x = data.processing()

    input_data = [[36, 0]]


    print(input_data)

import googleapiclient.discovery
# Fill in your PROJECT_ID, VERSION_NAME and MODEL_NAME before running
# this code.

PROJECT_ID = 'mypj-id'
VERSION_NAME = 'v1'
MODEL_NAME = 'model_xgb'

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
name += '/versions/{}'.format(VERSION_NAME)

response = service.projects().predict(
    name=name,
    body={'instances': input_data}
).execute()

if 'error' in response:
    print(response['error'])
else:
    online_results = response['predictions']

print(response)
print(online_results)

# print(online_results)

# %%
