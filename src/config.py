'''
@author: xusheng
'''

class ModelConfig(object):

    def __init__(self, model_name, input_data_dir, log_dir, batch_size=100, max_epoches=10, learning_rate=1e-3, fake_data=False):
        self.model_name = model_name
        self.input_data_dir = input_data_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.max_epoches = max_epoches
        self.learning_rate = learning_rate
        self.fake_data = fake_data
