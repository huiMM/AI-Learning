'''
@author: xusheng
'''
import tensorflow as tf
from model import Model
from config import ModelConfig
from six.moves import xrange

class ResnetModel(Model):
    def _conv(self, scope_name, x, filter_size, in_channels, out_channels, stride):
        with tf.variable_scope(scope_name):
            kernel = tf.get_variable(name='kernel', shape=[filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1))
            # backup tf.sqrt(2.0 / (filter_size * filter_size * out_channel))
            return tf.nn.conv2d(x, filter=kernel, strides=[1, stride, stride, 1], padding='SAME')

    def _batch_norm(self, scope_name, x):
        # ((x-mean) / var) * gamma + beta
        with tf.variable_scope(scope_name):
            params_shape = x.get_shape()[-1]
            # offset
            beta = tf.get_variable(name='beta', shape=[params_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
            # scale
            gamma = tf.get_variable(name='gamma', shape=[params_shape], dtype=tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

            mean, variance = tf.nn.moments(x, axes=[0, 1, 2], name='moments')

            bn = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=1e-3, name='batch_normal')
            bn.set_shape(x.get_shape())
#             tf.summary.histogram('bn', bn)
            return bn
    
    def _lrelu(self, x, alpha=0.0):
#         return tf.nn.leaky_relu(x, alpha=alpha, name='lrelu')
        return tf.nn.relu(x, name='relu')
    
    def _res_block(self, scope_name, x, in_channels, out_channels, stride):
        # Res: H(x) = F(x) + x
        # F(x) = Conv(Relu(BN( Conv(Relu(BN(x))) ))); bn->activation->conv
        with tf.variable_scope(scope_name):
            orig_x = x
            x = self._batch_norm('bn_1', x)
            x = self._lrelu(x)
            x = self._conv('conv_1', x, 3, in_channels, out_channels, stride)
            
            x = self._batch_norm('bn_2', x)
            x = self._lrelu(x)
            x = self._conv('conv_2', x, 3, out_channels, out_channels, 1)
            
            if in_channels != out_channels:
                orig_x = tf.nn.avg_pool(orig_x, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
                orig_x = tf.pad(orig_x,
                                [[0, 0],
                                 [0, 0],
                                 [0, 0],
                                 [(out_channels - in_channels) // 2, (out_channels - in_channels) // 2]
                                ])
            
            # H(x) = F(x) + x
            x += orig_x

#         tf.logging.debug('image after unit %s', x.get_shape())
        return x
    
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        # reduce dimension of 2nd & 3rd
        return tf.reduce_mean(x, [1, 2])

    def _fc(self, x, out_dim):
        x = tf.reshape(x, [self.config.batch_size, -1])
        w = tf.get_variable('weight', [x.get_shape()[1], out_dim], initializer=tf.initializers.variance_scaling(scale=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)
    
    def logits(self, data):
        # [batch, 28, 28, 1]
        x = tf.reshape(data, shape=[self.config.batch_size, self.config.input_size, self.config.input_size, 1])
        with tf.variable_scope('init_0'):
            # [batch, 28*28, 1, 16]
            x = self._conv('conv', x, filter_size=3, in_channels=1, out_channels=16, stride=1)
        
        # [batch, 28*28, 16, 32]
        x = self._res_block('block_1_0', x, in_channels=16, out_channels=32, stride=1)
        for i in xrange(1, 4):
            # [batch, 28*28, 32, 32]
            x = self._res_block('block_1_%d' % i, x, in_channels=32, out_channels=32, stride=1)
        
        # [batch, 14*14, 32, 64]
        x = self._res_block('block_2_0', x, in_channels=32, out_channels=64, stride=2)
        for i in xrange(1, 4):
            # [batch, 14*14, 64, 64]
            x = self._res_block('block_2_%d' % i, x, in_channels=64, out_channels=64, stride=1)
        
            # [batch, 7*7, 64, 128]
#         x = self._res_block('block_3_0', x, in_channels=64, out_channels=128, stride=2)
#         for i in xrange(1, 4):
            # [batch, 7*7, 128, 128]
#             x = self._res_block('block_3_%d' % i, x, in_channels=128, out_channels=128, stride=1)
        
        with tf.variable_scope('global_avgpool'):
            x = self._batch_norm('bn', x)
            x = self._lrelu(x)
            # [batch, 128]
            x = self._global_avg_pool(x)
        
        with tf.variable_scope('fc'):
            # [batch, 10]
            x = self._fc(x, self.config.num_classes)
        
        tf.summary.histogram('logits', x)
        
        return x


if __name__ == '__main__':
    m = ResnetModel(ModelConfig())
