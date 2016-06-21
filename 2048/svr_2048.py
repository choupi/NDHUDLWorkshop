from __future__ import print_function
import numpy as np
import math
np.random.seed(1337)  # for reproducibility

from sklearn.svm import SVR
from sklearn.metrics import *
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
X_train = X_train.reshape(X_train.shape[0], 4*4)
X_train = ll(X_train.astype('float32'))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_train = [float(y[0]) for y in y_train]

#print(X_train)
#print(Y_train)
print(X_train)
clf = SVR()
clf = clf.fit(X_train, Y_train)
y_predict = clf.predict(X_train)
print(y_predict)
print(mean_absolute_error(y_train, y_predict))
print(mean_squared_error(y_train, y_predict))

