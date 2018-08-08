'''
@author: xusheng
'''

import os
import tensorflow as tf
from model import Model
from tensorflow.examples.tutorials.mnist import input_data


class FaceModel(Model):

    def _fc(self, scope_name, x, output_size, stddev=1e-3):
        with tf.variable_scope(scope_name):
            x = tf.reshape(x, [self.config.batch_size, -1])
            w = tf.get_variable('weight', [x.get_shape()[-1], output_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('biases', [output_size], dtype=tf.float32, initializer=tf.constant_initializer(0., dtype=tf.float32))
            return tf.nn.xw_plus_b(x, w, b)
        
    def _conv(self, scope_name, x, filter_size, input_channels, output_channels, stride_size, padding='SAME', stddev=1e-1):
        with tf.variable_scope(scope_name):
            kernel = tf.get_variable('kernel', shape=[filter_size, filter_size, input_channels, output_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', shape=[output_channels], dtype=tf.float32, initializer=tf.constant_initializer(0., dtype=tf.float32))

            x = tf.nn.conv2d(x, filter=kernel, strides=[1, stride_size, stride_size, 1], padding=padding)
            x = tf.nn.bias_add(x, biases)
            return x

    def _trans_conv(self, scope_name, x, filter_size, output_shape, stride_size, padding='SAME', stddev=1e-3):
        with tf.variable_scope(scope_name):
            # x: [N, H, W, C]
            # output_shape: [height, width, input_channel, output_channel]
            # filter: [height, width, output_channels, in_channels]
            kernel = tf.get_variable('kernel', shape=[filter_size, filter_size, output_shape[-1], x.get_shape()[-1]], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', shape=[output_shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0., dtype=tf.float32))
            
            x = tf.nn.conv2d_transpose(x, filter=kernel, output_shape=output_shape, strides=[1, stride_size, stride_size, 1], padding=padding)
            x = tf.nn.bias_add(x, biases)
            return x

    def _lrelu(self, scope_name, x, alpha=0.0):
        with tf.variable_scope(scope_name):
            if alpha == 0.0:
                x = tf.nn.relu(x, name='relu')
            else:
                x = tf.nn.leaky_relu(x, alpha=alpha, name='lrelu')
            return x

    def _sigmoid(self, scope_name, x):
        with tf.variable_scope(scope_name):
            x = tf.nn.sigmoid(x, name='sigmoid')
        return x
    
    def _max_pool(self, scope_name, x, kernel_size, input_channels, output_channels, stride_size, padding='SAME', stddev=1e-1):
        with tf.variable_scope(scope_name):
            x = tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding, name='maxpool')
        return x

    def input_fn(self, data='mnist', mode='train', z=None):
        # raw data
        dataset = input_data.read_data_sets(os.path.join(self.config.input_data_dir, data), self.config.fake_data)
    
        # batch preparation
        if mode == 'train': # 55000
            tensors = (dataset.train.images, dataset.train.labels)
        elif mode == 'validation':  # 5000
            tensors = (dataset.validation.images, dataset.validation.labels)
        else:   # test 10000
            tensors = (dataset.test.images, dataset.test.labels)
        
        if z != None:
            pass
        
        ds = tf.data.Dataset.from_tensor_slices(tensors)
        ds_iter = ds.shuffle(buffer_size=10000).batch(self.config.batch_size).repeat(self.config.max_epoches).make_one_shot_iterator()
        
        return ds_iter
    
    def build_model(self, data):
        # roi [N, 128, 128, 3]
        input_data = tf.reshape(data, shape=[self.config.batch_size, self.config.input_size, self.config.input_size, self.config.channels])
        
        logits = self._conv('conv1', input_data, filter_size=5, input_channels=self.config.input_size*self.config.input_size*self.config.channels, output_channels=32, stride_size=1, padding='SAME', stddev=1e-3)
        logits = self._lrelu(logits)
        logits = self._
        with tf.variable_scope('conv1'):
            # [BATCH, 28, 28, 1] - conv2d(k=2, s=2, o=64), relu [BATCH, 14, 14, 1]
            # parameters: weight [2, 2, 1, 64], bias [64]
            kernel = tf.get_variable(name='weight', dtype=tf.float32, shape=[2, 2, self.config.channels, 64], initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
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
    
    def optimizer_fn(self, loss, global_step):
        op = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(loss, global_step=global_step)
        
        return op

    def eval_fn(self, logits, labels):
        eval_res = tf.nn.in_top_k(logits, tf.cast(labels, tf.int32), 1)
        correct_cnt = tf.reduce_sum(tf.cast(eval_res, tf.int32))
        
        return correct_cnt