'''
@author: xusheng
'''

import argparse
import os
import sys
import re

from six.moves import xrange
import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.basic_session_run_hooks import StepCounterHook


FLAGS = None

IMG_SIZE = 28

    
def main(_):
    with tf.Graph().as_default():
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
#             tf.summary('loss', loss)
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
        tf.summary.scalar('loss', loss)
        
        # global_step init
        global_step = tf.Variable(0, trainable=False, name='global_step')
        
        # optimizer
        training_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)
    
        # merge all summary defined
        merge_all = tf.summary.merge_all()
        
        # ckpt saver
        saver = tf.train.Saver()

        model_dir = "mnist-alpha"
        ckpt_name = "mnist-alpha"
        checkpoint_dir = os.path.join(FLAGS.log_dir, model_dir)
        
        # init summary writer
        writer = tf.summary.FileWriter(checkpoint_dir)
        
        init_op = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # restore checkpoint
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Load checkpoint")
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            else:
                print("Initialize checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(init_op)

            print("--- Begin loop ---")
            while True:
                try:
                    merge_all_summary, _ = sess.run([merge_all, training_op])
                    g_step = int(global_step.eval())
                    
                    # write summary
                    if g_step % 1 == 0:
                        writer.add_summary(merge_all_summary, g_step)
                        writer.flush()
                    # save ckpt
                    if g_step % 100 == 0:
                        saver.save(sess, os.path.join(checkpoint_dir, ckpt_name))
                        print("Save checkpoint @ global step %d" % g_step)
                except tf.errors.OutOfRangeError:
                    saver.save(sess, os.path.join(checkpoint_dir, ckpt_name))
                    print("Save checkpoint @ global step %d" % g_step)
                    writer.close()
                    print("--- End loop ---")
                    break

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
            default=1,
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
            default=os.path.join(os.getenv('TEST_TMPDIR', '..\\'), 'log'),
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
