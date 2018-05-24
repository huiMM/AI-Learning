'''
@author: xusheng
'''
import os
import tensorflow as tf
from model import Model
from resnet_model import ResnetModel
from resnet_bottleneck_model import ResnetBottleneckModel
from config import ModelConfig

def train(model):
    with tf.Graph().as_default() as g:
        ds_iter = model.input_fn(data='mnist', mode='train')
        
        # input data and label
        x_data, y_label = ds_iter.get_next()
        
        # global_step
        global_step = tf.Variable(0, trainable=False, name='global_step')
        
        # logits
        logits = model.logits(x_data)
        
        # loss
        loss = model.loss_fn(logits, y_label)
        
        # optimizer
        training_op = model.train_fn(loss, global_step)

        # merge all summary defined
        merge_all = tf.summary.merge_all()
        
        # ckpt saver
        saver = tf.train.Saver()

        checkpoint_dir = os.path.join(model.config.log_dir, model.config.model_name)
        
        # init summary writer
        writer = tf.summary.FileWriter(checkpoint_dir, graph=g)
        
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
                    if g_step % model.config.save_per_steps == 0:
                        saver.save(sess, os.path.join(checkpoint_dir, model.config.model_name))
                        print("Save checkpoint @ global step %d" % g_step)
                except tf.errors.OutOfRangeError:
                    saver.save(sess, os.path.join(checkpoint_dir, model.config.model_name))
                    print("Save checkpoint @ global step %d" % g_step)
                    writer.close()
                    print("--- End loop ---")
                    break

def trainDefaultModel():
    config = ModelConfig()
    config.model_name = 'mnist-alpha'
    config.input_data_dir = os.path.join('..', 'data')
    config.log_dir = os.path.join('..', 'log')
    config.batch_size = 100
    config.max_epoches = 2
    config.input_size = 28
    config.learning_rate = 1e-3
    config.num_classes = 10
    config.save_per_steps = 100
    config.fake_data = False
    
    model = Model(config)
    train(model)
    
def trainResnetModel():
    config = ModelConfig()
    config.model_name = 'mnist-resnet'
    config.input_data_dir = os.path.join('..', 'data')
    config.log_dir = os.path.join('..', 'log')
    config.batch_size = 100
    config.max_epoches = 1
    config.input_size = 28
    config.learning_rate = 1e-3
    config.num_classes = 10
    config.save_per_steps = 100
    config.fake_data = False
    
    model = ResnetModel(config)
    train(model)

def trainBottleneckResnetModel():
    config = ModelConfig()
    config.model_name = 'mnist-resnet-bottleneck'
    config.input_data_dir = os.path.join('..', 'data')
    config.log_dir = os.path.join('..', 'log')
    config.batch_size = 100
    config.max_epoches = 1
    config.input_size = 28
    config.learning_rate = 1e-3
    config.num_classes = 10
    config.save_per_steps = 10
    config.fake_data = False
    
    model = ResnetBottleneckModel(config)
    train(model)
    
def main(_):
    trainBottleneckResnetModel()

if __name__ == '__main__':
    tf.app.run(main=main)