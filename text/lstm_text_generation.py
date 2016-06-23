'''Example script to generate text from Nietzsche's writings.
'''

from __future__ import print_function
from keras.models import *
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

#chars = set(['\x86', '\n', '\xa9', '!', ' ', '"', '\xa4', "'", '\xa6', ')', '(', '\xab', '-', ',', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', ';', ':', '=', '?', '\xc3', '[', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'])
chars=set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
lenchars = len(chars)
sentences = []
next_chars = []

model = model_from_json(open('lstm_text_model.json').read())
model.load_weights('lstm_text_weights.h5')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
start_sentence = raw_input().strip()[:20]
for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    sentence = start_sentence
    generated = ''
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, lenchars))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
