from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_train = [int(y) for y in y_train]
Y_test = [int(y) for y in y_test]

clf = RandomForestClassifier(n_estimators=100, max_depth=3)
clf = clf.fit(X_train, y_train)

yt_predict = clf.predict_proba(X_train)
y_predict = clf.predict_proba(X_test)
print(log_loss(Y_train, yt_predict))
print(log_loss(Y_test, y_predict))
predict=[]
for yy in y_predict: predict.append(np.argmax(yy))
#print(predict)
print(f1_score(Y_test, predict))
print(classification_report(Y_test, predict))
print(confusion_matrix(Y_test, predict))
