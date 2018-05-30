'''
@author: xusheng
'''
import tensorflow as tf
from model import Model
from six.moves import xrange

class ResnetModel(Model):
    
    def _res_block(self, scope_name, x, in_channels, out_channels, stride=1):
        # Res: H(x) = F(x) + x
        # F(x) = Conv(Relu(BN( Conv(Relu(BN(x))) ))); bn->relu->conv
        with tf.variable_scope(scope_name):
            orig_x = x
            x = self._batch_norm('bn_1', x)
            x = self._lrelu(x)
            x = self._conv('conv_1', x, 3, in_channels, out_channels, stride)
            
            x = self._batch_norm('bn_2', x)
            x = self._lrelu(x)
            x = self._conv('conv_2', x, 3, out_channels, out_channels, 1)
            
            if in_channels != out_channels:
                if stride > 1:
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
    
    def build_model(self, data):
        # [batch, 28, 28, 1]
        x = tf.reshape(data, shape=[self.config.batch_size, self.config.input_size, self.config.input_size, self.config.channels])
        with tf.variable_scope('init_0'):
            # [batch, 28*28, 1, 16]
            x = self._conv('conv', x, filter_size=3, in_channels=self.config.channels, out_channels=16, stride=1)
        
        # [batch, 28*28, 16, 16]
        x = self._res_block('block_1_0', x, in_channels=16, out_channels=16, stride=1)
        for i in xrange(1, 4):
            # [batch, 28*28, 16, 16]
            x = self._res_block('block_1_%d' % i, x, in_channels=16, out_channels=16, stride=1)
        
        # [batch, 14*14, 16, 32]
        x = self._res_block('block_2_0', x, in_channels=16, out_channels=32, stride=2)
        for i in xrange(1, 4):
            # [batch, 14*14, 32, 32]
            x = self._res_block('block_2_%d' % i, x, in_channels=32, out_channels=32, stride=1)
        
        # [batch, 7*7, 32, 64]
        x = self._res_block('block_3_0', x, in_channels=32, out_channels=64, stride=2)
        for i in xrange(1, 4):
            # [batch, 7*7, 64, 64]
            x = self._res_block('block_3_%d' % i, x, in_channels=64, out_channels=64, stride=1)
        
        with tf.variable_scope('global_avgpool'):
            x = self._batch_norm('bn', x)
            x = self._lrelu(x)
            # [batch, 64]
            x = self._global_avg_pool(x)
        
        # [batch, 10]
        x = self._fc('fc', x, self.config.num_classes)
        
        tf.summary.histogram('logits', x)
        
        return x
