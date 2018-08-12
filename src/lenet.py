'''
@author: xusheng
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Reshape

class LeNet5(object):
    def __init__(self, name='LeNet5', weight_path=None):
        self._name = name
        self._weight_path = weight_path
    
    def build_model(self, row, col, channel, classes):
        model = Sequential()
        model.add(Reshape(target_shape=(row, col, channel), input_shape=(row*col*channel,)))
        model.add(Conv2D(filters=6, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        
        model.add(Conv2D(filters=16, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        
        model.add(Flatten())
        model.add(Dense(84))
        model.add(Activation('relu'))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        if self._weight_path is not None:
            model.load_weights(self._weight_path)
            
        self._model = model
        return self._model
    