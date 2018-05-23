'''
@author: xusheng
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from six.moves import xrange
import os

if __name__ == '__main__':
    with tf.variable_scope('sub1'):
        with tf.variable_scope('sub2'):
            a = tf.Variable(tf.random_normal(shape=[2, 2, 1, 2]))
            b = tf.reduce_mean(a, [1, 2])
            b = tf.reshape(b, [2, 1, 1, 2])
    
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        print(a.eval())
        print(b.eval())
