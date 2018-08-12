'''
@author: xusheng
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os

class Dataset(object):
    def __init__(self, **kwargs):
        self._dataset = None
        self._ds_train = None
        self._ds_val = None
        self._ds_test = None
        
        allowed_kwargs = {
            'z'
        }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

    def _get_ds_size(self, x):
        return x.shape[0]
    
    def load_dataset_train(self):
        return self._ds_train
    
    def load_dataset_val(self):
        return self._ds_val
    
    def load_dataset_test(self):
        return self._ds_test
    
    def get_ds_size_train(self):
        return self._get_ds_size(self._ds_train)
    
    def get_ds_size_val(self):
        return self._get_ds_size(self._ds_val)

    def get_ds_size_test(self):
        return self._get_ds_size(self._ds_test)

class MnistDataset(Dataset):
    def __init__(self, **kwargs):
        super(MnistDataset, self).__init__(**kwargs)
        
        self._dataset = 'mnist'
        self._raw_ds = input_data.read_data_sets(os.path.join('..', 'data', self._dataset), one_hot=True)
        self._ds_train = tf.data.Dataset.from_tensor_slices((self._raw_ds.train.images, self._raw_ds.train.labels))
        self._ds_val = tf.data.Dataset.from_tensor_slices((self._raw_ds.validation.images, self._raw_ds.validation.labels))
        self._ds_test = tf.data.Dataset.from_tensor_slices((self._raw_ds.test.images, self._raw_ds.test.labels))
    
    def get_ds_size_train(self):
        return self._get_ds_size(self._raw_ds.train.images)
    
    def get_ds_size_val(self):
        return self._get_ds_size(self._raw_ds.val.images)

    def get_ds_size_test(self):
        return self._get_ds_size(self._raw_ds.test.images)
