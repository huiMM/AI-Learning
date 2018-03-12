'''
@author: xusheng
'''

import argparse
import os
import sys
import re

# from six.moves import xrange
import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.basic_session_run_hooks import StepCounterHook

FLAGS = None

IMG_SIZE = 28

    
def main(_):
    # raw data
    dataset = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # batch preparation
    ds_training = tf.data.Dataset.from_tensor_slices((dataset.train.images, dataset.train.labels))
    ds_training = ds_training.shuffle(buffer_size=10000)
    ds_training = ds_training.batch(FLAGS.batch_size)
    ds_training = ds_training.repeat(FLAGS.max_epoches)
    iter = ds_training.make_one_shot_iterator()

    # logits
    x_data, y_label = iter.get_next()
    
    # [BATCH, 28, 28] - reshape [BATCH, 28, 28, 1]
    input = tf.reshape(x_data, shape=[FLAGS.batch_size, IMG_SIZE, IMG_SIZE, 1])
    
    with tf.variable_scope('conv1'):
        # [BATCH, 28, 28, 1] - conv2d(k=2, s=2, o=64), relu [BATCH, 14, 14, 1]
        # parameters: weight [2, 2, 1, 64], bias [64]
        kernel = tf.get_variable(name='weight', dtype=tf.float32, shape=[2, 2, 1, 64], initializer=tf.truncated_normal_initializer(mean=0., stddev=1.))
        bias = tf.get_variable(name='bias', dtype=tf.float32, shape=[64], initializer=tf.constant_initializer(0., dtype=tf.float32))
        conv = tf.nn.conv2d(input, filter=kernel, strides=[1, 2, 2, 1], padding='SAME')
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
    
    # loss: cross entropy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_label, tf.int32), logits=logits))
    
    # optimizer
    training_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    # initializer
    init_op = tf.global_variables_initializer()
    
    # ckpt saver
    saver = tf.train.Saver()
    
    model_name = "mnist-alpha.model"
    checkpoint_dir = os.path.join(FLAGS.log_dir, model_name)
    
    # run session
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, 
                                           hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_epoches*55000/FLAGS.batch_size/2)] 
#                                            save_checkpoint_secs=120, 
#                                            save_summaries_steps=100
                                           ) as mon_sess:
        print("0")
        mon_sess.run(init_op)
#         sess.run(tf.local_variables_initializer())
        
        global_step = 0
        
        # ckpt saver restore
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("1")
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(mon_sess, os.path.join(checkpoint_dir, ckpt_name))
            global_step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("Success to read %s, step starts from %d" % (ckpt_name, global_step))
        else:
            print("2")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)           
            
        while not mon_sess.should_stop():
            print("3")
            mon_sess.run(training_op)
            print("Step %d, loss = %s" % (global_step, mon_sess.run(loss)))        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=1e-3,
            help='Initial learning rate.'
    )
    parser.add_argument(
            '--max_epoches',
            type=int,
            default=2,
            help='Number of epoches to run trainer.'
    )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=100,
            help='Batch size.    Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
            '--input_data_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '..\\'), 'data\\mnist'),
            help='Directory to put the input data.'
    )
    parser.add_argument(
            '--log_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '..\\'), 'logs'),
            help='Directory to put the log data.'
    )
    parser.add_argument(
            '--fake_data',
            default=False,
            help='If true, uses fake data for unit testing.',
            action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
