'''
@author: xusheng
'''
import tensorflow as tf
from model import Model
from six.moves import xrange


class GanModel(Model):
    
    def _generator(self, z, y=None):
        # z: [N, z_dim]
        # g(z): z[N, z_dim] -> fc [N, 4*4*256] -> reshape [N, 4, 4, 256] -> deconv k5 s2 [N, 7, 7, 128]
        # -》 deconv k5 s2 [N, 14, 14, 64] -> deconv k5 s2 [N, 28, 28, 1]
        with tf.variable_scope('generator'):
            k_size = [_, 5, 5, 5]   # skip the first one
            stride_size = [_, 2, 2, 2]   # skip the first one
            layer_size = [4, 7, 14, 28]
            output_size = 64
            
            logits = self._fc('fc_0', z, layer_size[0] * layer_size[0] * output_size * 4)
            logits = tf.reshape(logits, [self.config.batch_size, layer_size[0], layer_size[0], output_size * 4])
            logits = self._batch_norm('bn_0', logits, epsilon=1e-5)
            logits = self._lrelu(logits)
            
            logits = self._trans_conv('deconv_1', logits, filter_size=k_size[1], output_shape=[self.config.batch_size, layer_size[1], layer_size[1], output_size * 2], stride_size=stride_size[1])
            logits = self._batch_norm('bn_1', logits, epsilon=1e-5)
            logits = self._lrelu(logits)

            logits = self._trans_conv('deconv_2', logits, filter_size=k_size[2], output_shape=[self.config.batch_size, layer_size[2], layer_size[2], output_size], stride_size=stride_size[2])
            logits = self._batch_norm('bn_2', logits, epsilon=1e-5)
            logits = self._lrelu(logits)

            logits = self._trans_conv('deconv_3', logits, filter_size=k_size[3], output_shape=[self.config.batch_size, layer_size[3], layer_size[3], self.config.channels], stride_size=stride_size[3])
            logits = self._batch_norm('bn_3', logits, epsilon=1e-5)
            logits = self._lrelu(logits)
            
            return logits

    def _discriminator(self, x, y=None, reuse=False):
        # x [N, 28, 28, 1]
        # D(x): x [N, 28, 28, 1] -> conv k3 s2 [N, 14, 14, 64] -> conv k3 s2 [N, 7, 7, 128]
        # -> conv k3 s2 [N, 4, 4, 256] -> reshape [N, 4*4*256] -> fc [N, 1]
        with tf.variable_scope("discriminator", reuse):
            k_size = [3, 3, 3]
            stride_size = [2, 2, 2]
            output_size = 64
            
            x = self._conv('conv_0', x, filter_size=k_size[0], input_channels=self.config.channels, output_channels=output_size, stride_size=stride_size[0])
            x = self._batch_norm('bn_0', x, epsilon=1e-5)
            x = self._lrelu(x, alpha=0.2)
            
            x = self._conv('conv_1', x, filter_size=k_size[1], input_channels=output_size, output_channels=output_size * 2, stride_size=stride_size[1])
            x = self._batch_norm('bn_1', x, epsilon=1e-5)
            x = self._lrelu(x, alpha=0.2)
            
            x = self._conv('conv_2', x, filter_size=k_size[2], input_channels=output_size * 2, output_channels=output_size * 4, stride_size=stride_size[2])
            x = self._batch_norm('bn_2', x, epsilon=1e-5)
            x = self._lrelu(x, alpha=0.2)
            
            x = tf.reshape(x, shape=[self.config.batch_size, -1])
            x = self._fc('fc', x, output_size=self.config.channels)

            return x
    
    def logits(self, data):
        # [BATCH, 28, 28] - reshape [BATCH, 28, 28, 1]
        input_data = tf.reshape(data, shape=[self.config.batch_size, self.config.input_size, self.config.input_size, self.config.channels])
        
        z_dim = 100
        z = tf.placeholder(dtype=tf.float32, shape=[self.conf.batch_size, z_dim], name='z')
        G = self._generator(z)
        
        # TODO
        
        
        
#         tf.summary.histogram('logits', logits)
        
#         return logits
