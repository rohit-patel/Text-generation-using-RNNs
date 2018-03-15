import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras

def window_transform_series(series, window_size):
    X = np.concatenate([[series[i:i+window_size] for i in range(len(series)-window_size)]])
    y = np.reshape(series[window_size:],(-1,1))
    return X,y

def build_part1_RNN(window_size):
    mod1 = Sequential()
    mod1.add(LSTM(units = 5, input_shape =(window_size,1)))
    mod1.add(Dense(1))
    return mod1

def cleaned_text(text):
    # punctuation = ['!', ',', '.', ':', ';', '?']
    import re
    return re.sub(' +', ' ', re.sub('(?![a-z!,.:;?]).+?',' ',text.replace('\n',' ').replace('\r',' ')))

def window_transform_text(text, window_size, step_size):
    inputs, outputs = zip(*[(text[i:i+window_size],text[i+window_size] ) for i in range(0,len(text)-window_size,step_size)])
    return list(inputs),list(outputs)

def build_part2_RNN(window_size, num_chars):
    mod1 = Sequential()
    mod1.add(LSTM(units = 21, input_shape =(window_size,num_chars),return_sequences=True ))
    mod1.add(LSTM(units = num_chars,return_sequences=False))
    mod1.add(Dense(1, activation='softmax'))
    return mod1