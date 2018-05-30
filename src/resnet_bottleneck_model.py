'''
@author: xusheng
'''
import tensorflow as tf
from resnet_model import ResnetModel
from six.moves import xrange

class ResnetBottleneckModel(ResnetModel):
    def __init__(self, config):
        self._config = config
        self._res_block = self._res_bottleneck_block
    
    def _res_bottleneck_block(self, scope_name, x, in_channels, out_channels, stride=1):
        with tf.variable_scope(scope_name):
            orig_x = x
            x = self._batch_norm('bn_1', x)
            x = self._lrelu(x)
            x = self._conv('conv_1', x, 1, in_channels, out_channels / 4, stride)
            
            x = self._batch_norm('bn_2', x)
            x = self._lrelu(x)
            x = self._conv('conv_2', x, 3, out_channels / 4, out_channels / 4, 1)
            
            x = self._batch_norm('bn_3', x)
            x = self._lrelu(x)
            x = self._conv('conv_3', x, 1, out_channels / 4, out_channels, 1)
            
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
            
            x = self._batch_norm('bn_4', x)
            x = self._lrelu(x)

#         tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def build_model(self, data):
        # [batch, 28, 28, 1]
        x = tf.reshape(data, shape=[self.config.batch_size, self.config.input_size, self.config.input_size, self.config.channels])
        with tf.variable_scope('init_0'):
            # [batch, 28*28, 1, 16]
            x = self._conv('conv', x, filter_size=3, in_channels=self.config.channels, out_channels=16, stride=1)
        
        # [batch, 28*28, 16, 64]
        x = self._res_block('block_1_0', x, in_channels=16, out_channels=64, stride=1)
        for i in xrange(1, 4):
            # [batch, 28*28, 64, 64]
            x = self._res_block('block_1_%d' % i, x, in_channels=64, out_channels=64, stride=1)
        
        # [batch, 14*14, 64, 256]
        x = self._res_block('block_2_0', x, in_channels=64, out_channels=256, stride=2)
        for i in xrange(1, 4):
            # [batch, 14*14, 256, 256]
            x = self._res_block('block_2_%d' % i, x, in_channels=256, out_channels=256, stride=1)
        
        # [batch, 7*7, 256, 1024]
        x = self._res_block('block_3_0', x, in_channels=256, out_channels=1024, stride=2)
        for i in xrange(1, 4):
            # [batch, 7*7, 1024, 1024]
            x = self._res_block('block_3_%d' % i, x, in_channels=1024, out_channels=1024, stride=1)
        
        with tf.variable_scope('global_avgpool'):
            x = self._batch_norm('bn', x)
            x = self._lrelu(x)
            # [batch, 1024]
            x = self._global_avg_pool(x)
        
        # [batch, 10]
        x = self._fc('fc', x, self.config.num_classes)
        
        tf.summary.histogram('logits', x)
        
        return x