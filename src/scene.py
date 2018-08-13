'''
@author: xusheng
'''

from models import LeNet5
from datasets import DatasetFactory
import os
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

class Scene(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {
            'max_epoch',
            'batch_size',
            'input_shape',
            'input.row',
            'input.col',
            'input.channel',
            'classes',
            'verbose',
            'load_weights',
            'ckpt_save_period',
            'log_path',
            'h5py'
        }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        
        load_weights = False
        if 'load_weights' in kwargs:
            load_weights = kwargs.pop('load_weights')

        # TODO: define ModelFactory to create specific model by name
        self._model = LeNet5(name='LeNet5', load_weight=load_weights, **kwargs).build_model()
        self._dataset = DatasetFactory().get_instance('mnist', **kwargs)

        if 'max_epoch' in kwargs:
            self._max_epoch = kwargs['max_epoch']

        if 'batch_size' in kwargs:
            self._batch_size = kwargs['batch_size']

        if 'input_shape' in kwargs:
            self._input_shape = kwargs['input_shape']

        if 'input.row' in kwargs and 'input.col' in kwargs and 'input.channel' in kwargs:
            self._row = kwargs['input.row']
            self._col = kwargs['input.col']
            self._channel = kwargs['input.channel']

        if 'classes' in kwargs:
            self._classes = kwargs['classes']

        if 'verbose' in kwargs:
            self._verbose = kwargs['verbose']

        if 'log_path' in kwargs:
            self._log_path = kwargs['log_path']

        if 'h5py' in kwargs:
            self._h5py = kwargs['h5py']
            
        if 'ckpt_save_period' in kwargs:
            self._ckpt_save_period = kwargs['ckpt_save_period']
        
    def train(self):
        ds_train = self._dataset.load_dataset_train()
        iter_train = ds_train.shuffle(buffer_size=10000).batch(batch_size=self._batch_size).repeat(count=self._max_epoch).make_one_shot_iterator()
        
        ds_val = self._dataset.load_dataset_val()
        iter_val = ds_val.shuffle(buffer_size=1000).batch(batch_size=self._batch_size).repeat(count=self._max_epoch).make_one_shot_iterator()
        
        ds_test = self._dataset.load_dataset_test()
        iter_test = ds_test.shuffle(buffer_size=2000).batch(batch_size=self._batch_size).make_one_shot_iterator()
        
        self._model.summary()
        
        optimizer = Adam()
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        ckptCallback = ModelCheckpoint(filepath=os.path.join(self._log_path, self._h5py), period=self._ckpt_save_period)
        logsCallback = TensorBoard(log_dir=self._log_path)
        
        history = self._model.fit(iter_train, None, epochs=self._max_epoch, steps_per_epoch=self._dataset.get_ds_size_train()//self._batch_size, verbose=self._verbose, validation_data=iter_val, validation_steps=self._dataset.get_ds_size_val()//self._batch_size, callbacks=[ckptCallback, logsCallback])
        
        score = self._model.evaluate(iter_test, None, verbose=self._verbose, steps=self._dataset.get_ds_size_test()//self._batch_size)
        print('Test score:', score[0])
        print('Test acurracy:', score[1])
        
        print(history.history.keys())
    
    def predict(self):
        ds_test = self._dataset.load_dataset_test()
        iter_test = ds_test.shuffle(buffer_size=2000).batch(batch_size=self._batch_size).make_one_shot_iterator()
        
        self._model.summary()
         
        optimizer = Adam()
        loss = 'categorical_crossentropy'
        self._model.compile(optimizer=optimizer, loss=loss)
         
#         out = self._model.predict(iter_test, verbose=self._verbose, steps=self._dataset.get_ds_size_test()//self._batch_size)
        out = self._model.predict(iter_test, verbose=self._verbose, steps=1)
        print(out)
        print('label:', np.argmax(out, axis=1))

def run_train():
    params = {
        'max_epoch': 200,
        'batch_size': 128,
        'input.row': 28,
        'input.col': 28,
        'input.channel': 1,
        'classes': 10,
        'verbose': 1,
        'log_path': os.path.join('..', 'log', 'mnist-lenet5'),
        'ckpt_save_period': 10, # # epoch
        
        # params for train
        'load_weights': False,
        'h5py': 'model-{epoch:02d}.h5'
    }
    scene = Scene(**params)
    scene.train()

def run_test():
    params = {
        'batch_size': 128,
        'input.row': 28,
        'input.col': 28,
        'input.channel': 1,
        'classes': 10,
        'verbose': 1,
        'log_path': os.path.join('..', 'log', 'mnist-lenet5'),
        'load_weights': True,
        'h5py': 'model-120.h5'  # a certain h5py config file
    }
    scene = Scene(**params)
    scene.predict()

if __name__ == '__main__':
#     run_train()
    run_test()
    