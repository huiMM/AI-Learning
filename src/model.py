'''
@author: xusheng
'''

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IMG_SIZE = 28

class Model(object):
    def __init__(self, config):
        self._config = config

    @ property
    def config(self):
        return self._config

    @ config.setter
    def config(self, value):
        self._config = value

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
        
#         ds = tf.data.Dataset.from_tensors(tensors)
#         ds_iter = ds.make_one_shot_iterator()
        
        return ds_iter
    
    def logits(self, data):
        # [BATCH, 28, 28] - reshape [BATCH, 28, 28, 1]
        input_data = tf.reshape(data, shape=[self.config.batch_size, IMG_SIZE, IMG_SIZE, 1])
        
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
            logits = tf.nn.bias_add(tf.matmul(logits, weight), bias)
            logits = tf.nn.relu(logits)
    
        with tf.variable_scope('fc2'):
            # [BATCH, 256] - linear, relu [BATCH, 10]
            # parameters: weight [256, 10], bias [10]
            weight = tf.get_variable(name='weight', dtype=tf.float32, shape=[256, 10], initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
            bias = tf.get_variable(name='bias', dtype=tf.float32, shape=[10], initializer=tf.constant_initializer(0., dtype=tf.float32))
            logits = tf.nn.bias_add(tf.matmul(logits, weight), bias)
        
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