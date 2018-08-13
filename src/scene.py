'''
@author: xusheng
'''

from lenet import LeNet5
from data import DatasetFactory, MnistDataset
import os

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
            'weight_path',
            'ckpt_path',
            'log_path'
        }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        
        # TODO: define ModelFactory to create specific model by name
        self._model = LeNet5(weight_path=None, **kwargs)
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
            
        if 'ckpt_path' in kwargs:
            self._ckpt_path = kwargs['ckpt_path']
            
        if 'log_path' in kwargs:
            self._log_path = kwargs['log_path']

    def train(self):
        ds_train = self._dataset.load_dataset_train()
        iter_train = ds_train.shuffle(buffer_size=10000).batch(batch_size=self._batch_size).repeat(count=self._max_epoch).make_one_shot_iterator()
        
        ds_val = self._dataset.load_dataset_val()
        iter_val = ds_val.shuffle(buffer_size=1000).batch(batch_size=self._batch_size).repeat(count=self._max_epoch).make_one_shot_iterator()
        
        ds_test = self._dataset.load_dataset_test()
        iter_test = ds_test.shuffle(buffer_size=2000).batch(batch_size=self._batch_size).repeat(count=1).make_one_shot_iterator()
        
        m = self._model.build_model()
        m.summary()
        
        optimizer = Adam()
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        m.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        ckptCallback = ModelCheckpoint(filepath=os.path.join(self._log_path, 'model-{epoch:02d}.h5'), period=1)
        logsCallback = TensorBoard(log_dir=self._log_path)
        
        history = m.fit(iter_train, None, epochs=self._max_epoch, steps_per_epoch=self._dataset.get_ds_size_train()//self._batch_size, verbose=self._verbose, validation_data=iter_val, validation_steps=self._dataset.get_ds_size_val()//self._batch_size, callbacks=[ckptCallback, logsCallback])
        
        score = m.evaluate(iter_test, None, verbose=self._verbose, steps=self._dataset.get_ds_size_test()//self._batch_size)
        print('Test score:', score[0])
        print('Test acurracy:', score[1])
        
        print(history.history.keys())

if __name__ == '__main__':
    params = {
        'max_epoch': 1,
        'batch_size': 128,
        'input.row': 28,
        'input.col': 28,
        'input.channel': 1,
        'classes': 10,
        'verbose': 1,
        'log_path': os.path.join('..', 'log', 'mnist-lenet5')
    }
    scene = Scene(**params)
    scene.train()
    