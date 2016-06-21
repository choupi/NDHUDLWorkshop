from __future__ import print_function
import numpy as np
import math
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

Y_train = y_train.astype('float32')

model = model_from_json(open('2048_model.json', 'rb').read())

model.load_weights('2048_weights.h5')
#score = model.evaluate(X_train, y_train, batch_size=16)
print(X_train)
score = model.predict(X_train, batch_size=1)
print(score)
