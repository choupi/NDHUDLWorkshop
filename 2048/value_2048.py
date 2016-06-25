from __future__ import print_function
import math
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import *
from keras.utils import np_utils
import csv
import sys

nb_train_samples = 48355

def load_data(csvf):
    X=np.zeros((nb_train_samples, 1, 4, 4), dtype="uint16")
    Y=[]
    i=0
    with open(csvf, 'rb') as f:
        for l in csv.reader(f):
            if len(l)<3: continue
            Y.append(int(l[0]))
            X[i,0,:,:] = np.reshape([int(j) for j in l[2:]], (4,4))
            i+=1
    Y=np.reshape(Y, (len(Y), 1))
    return (X, Y)

# the data, shuffled and split between train and test sets
(X_train, y_train) = load_data(sys.argv[1])

ll=np.vectorize(lambda x:math.log(x+1))
#X_train = X_train.reshape(X_train.shape[0], 1, 4, 4)
X_train = ll(X_train.astype('float32'))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_train = y_train.astype('float32')

model = Sequential()
model.add(BatchNormalization(axis=1, mode=2, input_shape=(1,4,4)))
#model.add(Convolution2D(256, nb_row=2, nb_col=2, input_shape=(1,4,4)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
#model.add(Activation('linear'))
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

open('value_2048_model.json', 'wb').write(model.to_json())
#print(X_train)
#print(Y_train)
model.fit(X_train, Y_train, nb_epoch=3, batch_size=1)
model.save_weights('value_2048_weights.h5', overwrite=True)
#score = model.evaluate(X_train, y_train, batch_size=16)
#print(X_train)
score = model.predict(X_train, batch_size=1)
print(score)

