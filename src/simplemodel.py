'''
@author: xusheng
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from six.moves import xrange
from _testbuffer import ndarray

if __name__ == '__main__':
    
#     x_vals = np.random.normal(1, 0.1, 100)
#     y_vals = np.repeat(10., 100)
#     
#     x_data = tf.placeholder(tf.float32, shape=[None, 1])
#     y_target = tf.placeholder(tf.float32, shape=[None, 1])
#     
#     A = tf.Variable(tf.random_normal([1, 1]))
#     
#     my_output = tf.matmul(x_data, A)
#     
#     loss = tf.nn.l2_loss(my_output - y_target)
# 
#     opt = tf.train.GradientDescentOptimizer(0.02)
#     
#     train_step = opt.minimize(loss)
#     
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         
#         loss_batch = []
#         for i in xrange(100):
#             rand_index = np.random.choice(100, size=BATCH_SIZE)
#             rand_x = np.transpose([x_vals[rand_index]])
#             rand_y = np.transpose([y_vals[rand_index]])
#             sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
#             if (i+1)%1 == 0:
#                 print('Step #%d: A = %s' % ((i + 1), sess.run(A)))
#                 loss_eval = sess.run(loss, feed_dict={x_data:rand_x, y_target: rand_y})
#                 print('Loss = %s' % loss_eval)
#                 loss_batch.append(loss_eval)
# 
#         plt.plot(range(0, 100, 1), loss_batch, 'r-', label='Batch Loss, size=20')
#         plt.legend(loc='upper right', prop={'size': 11})
#         plt.show()
    
    BATCH_SIZE = 20
    EPOCHES = 1000
    
    iris = datasets.load_iris()
#     binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    a = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1]))

    inference = tf.add(tf.matmul(x_data, a), b)

    loss = tf.nn.l2_loss(inference - y_target)

    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Training
        loss_batch = []
        for i in xrange(EPOCHES):
            rand_index = np.random.choice(len(iris.data), size=BATCH_SIZE)
            
            rand_x = np.reshape(iris.data[rand_index, 3], [BATCH_SIZE, 1])
            rand_y = np.reshape(iris.data[rand_index, 0], [BATCH_SIZE, 1])
            
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            if (i+1)%1==0:
                l = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
                loss_batch.append(l)
#                 print('Step #%d， A = %s, b = %s, loss = %s' % ((i+1), sess.run(a), sess.run(b), l))
    
        # Testing

        # Diagram        
        plt.plot(range(0, EPOCHES, 1), loss_batch, 'r-', label='L2 loss per epoches')
        plt.legend(loc='upper right', prop={'size': 11})
        plt.show()



    