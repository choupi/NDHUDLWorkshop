'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
import xgboost as xgb
from sklearn.metrics import *

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_train = [int(y) for y in y_train]
Y_test = [int(y) for y in y_test]

param = {}
param['objective'] = 'multi:softprob'
param['booster'] = 'gbtree'
param['num_class'] = 10
param['eta'] = 0.1
param['max_depth'] = 4
param['min_child_weight'] = 0.316
param['silent'] = 1
plst = list(param.items())
num_round = 12
model_file='mnist.model'

xgmat = xgb.DMatrix(X_train, label=Y_train)
watchlist = [ (xgmat,'train') ]
bst = xgb.train(plst, xgmat, num_round, watchlist)
bst.save_model(model_file)

xgmat = xgb.DMatrix(X_test)
bst = xgb.Booster(model_file = model_file)
y_predict = bst.predict(xgmat)
#y_predict = bst.predict_proba(xgmat)
print(y_predict)

print(log_loss(Y_test, y_predict))
#print(y_predict)
#predict=y_predict
predict=[]
for yy in y_predict: predict.append(np.argmax(yy))
#print(predict)
print(f1_score(Y_test, predict))
print(classification_report(Y_test, predict))
print(confusion_matrix(Y_test, predict))
