import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense  
from tensorflow.keras.optimizers import Adam
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

### Regression models ###

monitor = EarlyStopping(
        monitor='loss', min_delta=100000, 
        patience=10, verbose=1, mode='auto',
        restore_best_weights=True
)

def shallow_mlp(Xtrn, Ytrn):
    model_mlp = Sequential()
    model_mlp.add(Dense(100, activation='relu', input_dim=Xtrn.shape[1]))
    model_mlp.add(Dense(1))
    model_mlp.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model_mlp.fit(Xtrn, Ytrn, epochs=50, batch_size=10000, callbacks=[monitor])

    return model_mlp

def deep_mlp(Xtrn, Ytrn):
    model_mlp = Sequential()
    model_mlp.add(Dense(100, activation='relu', input_dim=Xtrn.shape[1]))
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(Dense(1))
    model_mlp.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model_mlp.fit(Xtrn, Ytrn, epochs=50, batch_size=10000, callbacks=[monitor])

    return model_mlp

def deepest_mlp(Xtrn, Ytrn):
    model_mlp = Sequential()
    model_mlp.add(Dense(100, activation='relu', input_dim=Xtrn.shape[1]))
    model_mlp.add(Dense(100, activation='relu'))
    model_mlp.add(Dense(100, activation='relu'))
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(Dense(1))
    model_mlp.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model_mlp.fit(Xtrn, Ytrn, epochs=50, batch_size=10000, callbacks=[monitor])

    return model_mlp

def cnn_regression(Xtrn, Ytrn, look_back):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, Xtrn.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear')) 
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(Xtrn, Ytrn, epochs=50, batch_size=10000, verbose=2, callbacks=[monitor])

    return model

def lstm_regression(Xtrn, Ytrn, look_back):
    model = Sequential()
    model.add(LSTM(256,input_shape=(look_back, Xtrn.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    model.fit(Xtrn, Ytrn, epochs=2, batch_size=40000, verbose=2, callbacks=[monitor])

    return model

### Classification model ###
def classification_model(Xtrn, Ytrn):
    model_mlp = Sequential()
    model_mlp.add(Dense(100, input_dim=Xtrn.shape[1], activation='relu'))
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(Dense(3, activation='softmax'))
    model_mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_mlp.fit(Xtrn, Ytrn, epochs=50, batch_size=10000)

    return model_mlp