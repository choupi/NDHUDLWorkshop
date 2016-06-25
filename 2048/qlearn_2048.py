from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from qlearning4k import Agent
from my2048 import mGame

grid_size = 4
nb_frames = 1
nb_actions = 4

model = Sequential()
model.add(BatchNormalization(axis=1, mode=2, input_shape=(nb_frames, grid_size, grid_size)))
#model.add(Convolution2D(256, nb_row=4, nb_col=4, input_shape=(nb_frames, grid_size, grid_size)))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, nb_row=2, nb_col=2, activation='relu'))
#model.add(Convolution2D(32, nb_row=4, nb_col=4, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(nb_actions))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
#model.compile(RMSprop(), 'MSE')
open('2048_model.json', 'wb').write(model.to_json())
print model.input_shape

m2048 = mGame()
print m2048.get_frame().shape

agent = Agent(model=model, memory_size=65536, nb_frames=nb_frames)
agent.train(m2048, batch_size=512, nb_epoch=10, epsilon=(.9,.05), gamma=0.1)
model.save_weights('2048_weights.h5', overwrite=True)

#agent.play(m2048)
