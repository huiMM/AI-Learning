'''
@author: xusheng
'''

class ModelConfig(object):
    @property
    def model_name(self):
        return self._model_name
    
    @model_name.setter
    def model_name(self, value):
        self._model_name = value
        
    @property
    def input_data_dir(self):
        return self._input_data_dir
    
    @input_data_dir.setter
    def input_data_dir(self, value):
        self._input_data_dir = value

    @property
    def log_dir(self):
        return self._log_dir
    
    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value
        
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def max_epoches(self):
        return self._max_epoches
    
    @max_epoches.setter
    def max_epoches(self, value):
        self._max_epoches = value
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, value):
        self._input_size = value
        
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        
    @property
    def fake_data(self):
        return self._fake_data
    
    @fake_data.setter
    def fake_data(self, value):
        self._fake_data = value