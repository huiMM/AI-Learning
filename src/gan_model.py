'''
@author: xusheng
'''
import tensorflow as tf
from model import Model
from six.moves import xrange


class GanModel(Model):

    def _generator(self, z, y=None):
        # shape of z: [batch, z_dim]
        z_shape = z.get_shape().as_list()

        with tf.variable_scope("generator") as scope:
            weight = tf.get_variable(name='weight', dtype=tf.float32, shape=[z_shape[1], 64*8*4*4], initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02))
            bias = tf.get_variable(name='bias', dtype=tf.float32, shape=[64*8*4*4], initializer=tf.constant_initializer(0., dtype=tf.float32))
            # [batch, 64*8*4*4]
            logits = tf.nn.bias_add(tf.matmul(z, weight), bias)
            
            logits = tf.reshape(logits, [-1, 4, 4, 64*8])
            logits = tf.nn.relu(self._batch_norm(scope, logits, epsilon=1e-5))

    
    def _discriminator(self, x, y=None, reuse=False):
        pass
    
    def logits(self, data):
        self._generator()
