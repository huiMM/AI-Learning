'''
@author: xusheng
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Reshape
import os

import numpy as np

class Model(object):
    def __init__(self, name, load_weights=False, **kwargs):
        self._name = name
        self._load_weights = load_weights
        
        self._input_shape = None
        self._row = None
        self._col = None
        self._channel = None
        
        allowed_kwargs = {
            'log_path',
            'h5py',
            'input_shape',
            'input.row',
            'input.col',
            'input.channel',
            'classes'
        }
#         for kwarg in kwargs:
#             if kwarg not in allowed_kwargs:
#                 raise TypeError('Keyword argument not understood:', kwarg)
        
        if 'input_shape' in kwargs:
            self._input_shape = kwargs['input_shape']
        
        if 'input.row' in kwargs and 'input.col' in kwargs and 'input.channel' in kwargs:
            self._row = kwargs['input.row']
            self._col = kwargs['input.col']
            self._channel = kwargs['input.channel']
            
        if 'classes' in kwargs:
            self._classes = kwargs['classes']
        
        if 'log_path' in kwargs:
            self._log_path = kwargs['log_path']
            
        if 'h5py' in kwargs:
            self._h5py = kwargs['h5py']
    
    def _build_model(self):
        return None
    
    def build_model(self):
        model = self._build_model()
        if self._load_weights == True:
            model.load_weights(os.path.join(self._log_path, self._h5py))
        model.summary()
        self._model = model
        return self._model
    
class LeNet5(Model):
    def __init__(self, name='LeNet5', load_weights=False, **kwargs):
        super(LeNet5, self).__init__(name=name, load_weights=load_weights, **kwargs)
    
    def _build_model(self):
        model = Sequential()
        if self._input_shape is None:
            model.add(Reshape(target_shape=(self._row, self._col, self._channel), input_shape=(self._row * self._col * self._channel,)))
        else:
            # TODO: _input_shape as input
            pass
        model.add(Conv2D(filters=6, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        
        model.add(Conv2D(filters=16, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        
        model.add(Flatten())
        model.add(Dense(84))
        model.add(Activation('relu'))
        
        model.add(Dense(self._classes))
        model.add(Activation('softmax'))
        
#         for layer in model.layers:
#             weight = layer.get_weights()
#             print(layer, np.asarray(weight).shape)
        return model

# class DCGAN(Model):
#     def __init__(self, name='DCGAN', load_weights=False, **kwargs):
#         super(DCGAN, self).__init__(name=name, load_weights=load_weights, **kwargs)
# 
#     def _build_model(self):
#         model = Sequential()
#         if self._input_shape is None:
#             model.add(Reshape(target_shape=(self._row, self._col, self._channel), input_shape=(self._row * self._col * self._channel,)))
#         else:
#             # TODO: _input_shape as input
#             pass
#         model.add(Conv2D(filters=6, kernel_size=5, padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPool2D(pool_size=2, strides=2))
#         
#         model.add(Conv2D(filters=16, kernel_size=5, padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPool2D(pool_size=2, strides=2))
#         
#         model.add(Flatten())
#         model.add(Dense(84))
#         model.add(Activation('relu'))
#         
#         model.add(Dense(self._classes))
#         model.add(Activation('softmax'))
#         return model