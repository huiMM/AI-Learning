'''
@author: xusheng
'''

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Model(object):
    def __init__(self, config):
        self._config = config

    @ property
    def config(self):
        return self._config

    @ config.setter
    def config(self, value):
        self._config = value

    def _conv(self, scope_name, x, filter_size, in_channels, out_channels, stride):
        with tf.variable_scope(scope_name):
            kernel = tf.get_variable(name='kernel', shape=[filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1))
            # backup tf.sqrt(2.0 / (filter_size * filter_size * out_channel))
            return tf.nn.conv2d(x, filter=kernel, strides=[1, stride, stride, 1], padding='SAME')

    def _batch_norm(self, scope_name, x, scale=1.0, offset=0.0, epsilon=1e-3):
        # x_norm =  gamma * ((x - x_mean) / sqrt(x_var + epsilon) + beta
        
        # running_x_mean = momentum * running_x_mean + (1 - momentum) * x_mean
        # running_x_var = momentum * running_x_var + (1 - momentum) * x_var
        # replace BN implementation with tf.contrib.layers.batch_norm, decay = momentum, usually 0.999, 0.99, 0.9 etc
        with tf.variable_scope(scope_name):
            # out_channels' shape
            params_shape = x.get_shape()[-1]
            # offset
            beta = tf.get_variable(name='beta', shape=[params_shape], dtype=tf.float32, initializer=tf.constant_initializer(offset, tf.float32))
            # scale
            gamma = tf.get_variable(name='gamma', shape=[params_shape], dtype=tf.float32, initializer=tf.constant_initializer(scale, tf.float32))

            # axes [0, 1, 2] for [batch, height, width, out_channels]
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2], name='moments')

            bn = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=epsilon, name='batch_normal')
            bn.set_shape(x.get_shape())
#             tf.summary.histogram('bn', bn)
            return bn
    
    def _lrelu(self, x, alpha=0.0):
#         return tf.nn.leaky_relu(x, alpha=alpha, name='lrelu')
        return tf.nn.relu(x, name='relu')
    
    def input_fn(self, data='mnist', mode='train'):
        # raw data
        dataset = input_data.read_data_sets(os.path.join(self.config.input_data_dir, data), self.config.fake_data)
    
        # batch preparation
        if mode == 'train': # 55000
            tensors = (dataset.train.images, dataset.train.labels)
        elif mode == 'validation':  # 5000
            tensors = (dataset.validation.images, dataset.validation.labels)
        else:   # test 10000
            tensors = (dataset.test.images, dataset.test.labels)
        
        ds = tf.data.Dataset.from_tensor_slices(tensors)
        ds_iter = ds.shuffle(buffer_size=10000).batch(self.config.batch_size).repeat(self.config.max_epoches).make_one_shot_iterator()
        
        return ds_iter
    
    def logits(self, data):
        # [BATCH, 28, 28] - reshape [BATCH, 28, 28, 1]
        input_data = tf.reshape(data, shape=[self.config.batch_size, self.config.input_size, self.config.input_size, 1])
        
        with tf.variable_scope('conv1'):
            # [BATCH, 28, 28, 1] - conv2d(k=2, s=2, o=64), relu [BATCH, 14, 14, 1]
            # parameters: weight [2, 2, 1, 64], bias [64]
            kernel = tf.get_variable(name='weight', dtype=tf.float32, shape=[2, 2, 1, 64], initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
    #             tf.summary('loss', loss)
            bias = tf.get_variable(name='bias', dtype=tf.float32, shape=[64], initializer=tf.constant_initializer(0., dtype=tf.float32))
            conv = tf.nn.conv2d(input_data, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
            logits = tf.nn.bias_add(conv, bias)
            logits = tf.nn.relu(logits)
            
        with tf.variable_scope('pool1'):
            # [BATCH, 14, 14, 1] - maxpool(k=2, s=2) [BATCH, 7, 7, 1]
            logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
        with tf.variable_scope('fc1'):
            # [BATCH, 7, 7, 1] - reshape [BATCH, 64*7*7] - linear, relu [BATCH, 256]
            # parameters: weight [64*7*7, 256], bias [256]
            logits = tf.reshape(logits, shape=[-1, 64*7*7])
            weight = tf.get_variable(name='weight', dtype=tf.float32, shape=[64*7*7, 256], initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
            bias = tf.get_variable(name='bias', dtype=tf.float32, shape=[256], initializer=tf.constant_initializer(0., dtype=tf.float32))
            logits = tf.nn.xw_plus_b(logits, weight, bias)
            logits = tf.nn.relu(logits)
    
        with tf.variable_scope('fc2'):
            # [BATCH, 256] - linear, relu [BATCH, 10]
            # parameters: weight [256, 10], bias [10]
            weight = tf.get_variable(name='weight', dtype=tf.float32, shape=[256, self.config.num_classes], initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
            bias = tf.get_variable(name='bias', dtype=tf.float32, shape=[self.config.num_classes], initializer=tf.constant_initializer(0., dtype=tf.float32))
            logits = tf.nn.xw_plus_b(logits, weight, bias)
        
        tf.summary.histogram('logits', logits)
        
        return logits
    
    def loss_fn(self, logits, labels):
        # loss: cross entropy
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits))
        tf.summary.scalar('loss', loss_op)
        
        return loss_op
    
    def train_fn(self, loss, global_step):
        train_op = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(loss, global_step=global_step)
        
        return train_op

    def eval_fn(self, logits, labels):
        eval_res = tf.nn.in_top_k(logits, tf.cast(labels, tf.int32), 1)
        correct_cnt = tf.reduce_sum(tf.cast(eval_res, tf.int32))
        
        return correct_cnt