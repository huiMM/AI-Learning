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
    BATCH_SIZE = 20
    EPOCHES = 20000
    NUM_CAT = 3
    
    iris = datasets.load_iris()
#     binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
    x_data = tf.placeholder(shape=[BATCH_SIZE, 4], dtype=tf.float32)
    y_target = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32)

    a = tf.Variable(tf.random_normal(shape=[4, NUM_CAT]))
    b = tf.Variable(tf.random_normal(shape=[NUM_CAT]))

    inference = tf.reshape(tf.add(tf.matmul(x_data, a), b), [BATCH_SIZE, NUM_CAT])

#     alpha = 1.
#     lamb = 1.
#     a_l1_loss = tf.reduce_mean(tf.abs(a))
#     a_l2_loss = tf.reduce_mean(tf.square(a))
#     loss = tf.nn.l2_loss(inference - y_target) + lamb * ((1-alpha) * a_l2_loss + alpha * a_l1_loss) 
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=inference))
    tf.summary.scalar('loss', loss)
    
    global_step = tf.train.get_or_create_global_step()
    
    train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss, global_step=global_step)
    
    
    class _LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._step = 0
        
        def before_run(self, run_context):
            rand_index = np.random.choice(len(iris.data), size=BATCH_SIZE)
            rand_x = np.reshape(iris.data[rand_index, :], [BATCH_SIZE, 4])
            rand_y = iris.target[rand_index]
            return tf.train.SessionRunArgs(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    
        def after_run(self, run_context, run_values):
            self._step += 1
            print('step = %d, loss = %s' % (self._step, run_values.results))

    # run session
    with tf.train.MonitoredTrainingSession(checkpoint_dir='..\\log\\test', 
                                            hooks=[tf.train.StopAtStepHook(num_steps=10000), 
                                                   tf.train.NanTensorHook(loss), 
                                                    _LoggerHook()], 
                                            save_checkpoint_secs=120, 
                                            save_summaries_steps=100
                                           ) as mon_sess:
            
        while not mon_sess.should_stop():
#             print("before mon_sess run")
            mon_sess.run(train_op)
#             print("after mon_sess run")
#             print("Step %s, loss = %s" % (mon_sess.run(global_step), mon_sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
# 
#         # Training
#         total_loss_batch = []
# #         a_l1_loss_batch = []
# #         a_l2_loss_batch = []
#         for i in xrange(EPOCHES):
#             rand_index = np.random.choice(len(iris.data), size=BATCH_SIZE)
#             
#             rand_x = np.reshape(iris.data[rand_index, :], [BATCH_SIZE, 4])
#             rand_y = iris.target[rand_index]
#             
#             sess.run(train_op, feed_dict={x_data: rand_x, y_target: rand_y})
#             
#             epoch_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
#             total_loss_batch.append(epoch_loss)
#             if (i+1)%100==0:
# #                 a_l1_l = sess.run(a_l1_loss, feed_dict={x_data:rand_x})
# #                 a_l1_loss_batch.append(a_l1_l)
# #                 a_l2_l = sess.run(a_l2_loss, feed_dict={x_data:rand_x})
# #                 a_l2_loss_batch.append(a_l2_l)
#                 print('Step #%d， A = %s, b = %s, loss = %s' % ((i+1), sess.run(a), sess.run(b), epoch_loss))
# #                 print('Step #%d， loss = %s' % ((i+1), l))
    
        # Testing

        # Diagram
#         plt.plot(range(0, EPOCHES, 1), total_loss_batch, 'b-', label='Total Loss')
#         plt.plot(range(0, EPOCHES, 1), a_l1_loss_batch, 'b--', label='Alaph loss l1 per epoches')
#         plt.plot(range(0, EPOCHES, 1), a_l2_loss_batch, 'g--', label='Alaph loss l2 per epoches')
#         plt.legend(loc='upper right', prop={'size': 10})
#         plt.show()



    