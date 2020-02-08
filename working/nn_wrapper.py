# %%
import pdb
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout


class NNWrapper(object):
    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        self.scaler.fit(tr_x)
        # MinMaxScaler(copy=True, feature_range=(0, 1))
        tr_x_scaled = self.scaler.transform(tr_x)
        va_x_scaled = self.scaler.transform(va_x)

        model = Sequential()
        model.add(Dense(256, activation='relu',
                        input_shape=(tr_x_scaled.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')

        batch_size = 128
        epochs = 10

        history = model.fit(tr_x_scaled,
                            tr_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(va_x_scaled, va_y))

        # with open(f'../output/nn_model.pickle', 'wb') as f:
        # pickle.dump(model, f)
        # model.save('../output/nn_model.h5')

        self.model = model

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict_proba(x_scaled).reshape(-1)
        return pred


# %%
