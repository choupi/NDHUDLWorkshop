from keras.models import Sequential,model_from_json
from keras.layers import *
from keras.optimizers import *
from qlearning4k import Agent
from my2048 import mGame

grid_size = 4
nb_frames = 1
nb_actions = 4

model = model_from_json(open('2048_model.json', 'rb').read())
print model.input_shape

m2048 = mGame(vis=True)
#m2048 = mGame()
print m2048.get_frame().shape

agent = Agent(model=model, memory_size=65536, nb_frames=nb_frames)
model.load_weights('2048_weights.h5')
model.summary()
print model.get_weights()

agent.play(m2048, epsilon=0.1, visualize=False)
