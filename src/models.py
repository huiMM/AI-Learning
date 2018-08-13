'''
@author: xusheng
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Reshape

class Model(object):
    def __init__(self, name, weight_path=None, **kwargs):
        self._name = name
        self._weight_path = weight_path
        
        self._input_shape = None
        self._row = None
        self._col = None
        self._channel = None
        
        allowed_kwargs = {
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
    
    def _build_model(self):
        return None
    
    def build_model(self):
        model = self._build_model()
        if self._weight_path is not None:
            model.load_weights(self._weight_path)
        
        self._model = model
        return self._model
    
class LeNet5(Model):
    def __init__(self, name='LeNet5', weight_path=None, **kwargs):
        super(LeNet5, self).__init__(name=name, weight_path=weight_path, **kwargs)
    
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
        return model
    