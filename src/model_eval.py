'''
@author: xusheng
'''
import os
import tensorflow as tf
from config import ModelConfig
from model import Model
from resnet_model import ResnetModel


def evaluate(model):
    with tf.Graph().as_default():
        ds_iter = model.input_fn(data='mnist', mode='test')
        
        # input data and label
        x_data, y_label = ds_iter.get_next()
        
        # global_step
        global_step = tf.Variable(0, trainable=False, name='global_step')
        
        # logits
        logits = model.logits(x_data)
        
        correct_cnt = model.eval_fn(logits, y_label)
        
        # ckpt saver
        saver = tf.train.Saver()

        checkpoint_dir = os.path.join(model.config.log_dir, model.config.model_name)
        
        with tf.Session() as sess:
            # restore checkpoint
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Load checkpoint")
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
                
                while True:
                    try:
                        print("Load checkpoint @ global step %d, accuracy = %2f%%" % (global_step.eval(), correct_cnt.eval() * 100.0 / 10000.0))
                    except tf.errors.OutOfRangeError:
                        break
            else:
                print("Fail to load checkpoint")


def evaluateDefaultModel():
    config = ModelConfig()
    config.model_name = 'mnist-alpha'
    config.input_data_dir = os.path.join('..', 'data')
    config.log_dir = os.path.join('..', 'log')
    config.batch_size = 10000
    config.max_epoches = 1
    config.input_size = 28
    config.num_classes = 10
    config.fake_data = False
    
    model = Model(config)
    evaluate(model)


def evaluateResnetModel():
    config = ModelConfig()
    config.model_name = 'mnist-resnet'
    config.input_data_dir = os.path.join('..', 'data')
    config.log_dir = os.path.join('..', 'log')
    config.batch_size = 10000
    config.max_epoches = 1
    config.input_size = 28
    config.num_classes = 10
    config.fake_data = False
    
    model = ResnetModel(config)
    evaluate(model)

    
def main(_):
    evaluateResnetModel()


if __name__ == '__main__':
    tf.app.run(main=main)
