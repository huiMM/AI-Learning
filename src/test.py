'''
@author: xusheng
'''
from lenet import LeNet5
from data import MnistDataset
import os
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

def train():
    NB_EPOCH = 20
    NB_CLASSES = 10
    BATCH_SIZE = 128
    VERBOSE = 2
    IMG_SIZE = 28
    CHANNEL_SIZE = 1
#     INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNEL_SIZE)
#     INPUT_SHAPE = (IMG_SIZE * IMG_SIZE * CHANNEL_SIZE)
    
    mnist = MnistDataset()
    ds_train = mnist.load_dataset_train()
    iter_train = ds_train.shuffle(buffer_size=10000).batch(batch_size=BATCH_SIZE).repeat(count=NB_EPOCH).make_one_shot_iterator()
    
    ds_val = mnist.load_dataset_val()
    iter_val = ds_val.shuffle(buffer_size=1000).batch(batch_size=BATCH_SIZE).repeat(count=NB_EPOCH).make_one_shot_iterator()
    
    ds_test = mnist.load_dataset_test()
    iter_test = ds_test.shuffle(buffer_size=2000).batch(batch_size=BATCH_SIZE).repeat(count=1).make_one_shot_iterator()
    
    model = LeNet5().build_model(row=IMG_SIZE, col=IMG_SIZE, channel=CHANNEL_SIZE, classes=NB_CLASSES)
    model.summary()
    
    optimizer = Adam()
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    ckptCallback = ModelCheckpoint(filepath=os.path.join('..', 'log', 'mnist-lenet5', 'model-{epoch:02d}.h5'), period=10)
    logsCallback = TensorBoard(log_dir=os.path.join('..', 'log', 'mnist-lenet5'))
    
    history = model.fit(iter_train, None, epochs=NB_EPOCH, steps_per_epoch=mnist.get_ds_size_train()//BATCH_SIZE, verbose=VERBOSE, validation_data=iter_val, validation_steps=mnist.get_ds_size_val()//BATCH_SIZE, callbacks=[ckptCallback, logsCallback])
    
    score = model.evaluate(iter_test, None, verbose=VERBOSE, steps=mnist.get_ds_size_test()//BATCH_SIZE)
    print('Test score:', score[0])
    print('Test acurracy:', score[1])
    
    print(history.history.keys())

def predict():
    NB_CLASSES = 10
#     BATCH_SIZE = 128
    VERBOSE = 1
    IMG_SIZE = 28
    CHANNEL_SIZE = 1
     
    mnist = MnistDataset()
    ds_test = mnist.load_dataset_test()
#     iter_test = ds_test.shuffle(buffer_size=2000).batch(batch_size=BATCH_SIZE).make_one_shot_iterator()
    iter_test = ds_test.shuffle(buffer_size=2000).batch(batch_size=1).make_one_shot_iterator()
     
    model = LeNet5(os.path.join('..', 'log', 'mnist-lenet5', 'model-20.h5')).build_model(row=IMG_SIZE, col=IMG_SIZE, channel=CHANNEL_SIZE, classes=NB_CLASSES)
     
    optimizer = Adam()
    loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss)
     
#     out = model.predict(iter_test, verbose=VERBOSE, steps=5000//BATCH_SIZE)
    out = model.predict(iter_test, verbose=VERBOSE, steps=1)
    print(out)
    print('label:', np.argmax(out, axis=1))
     

if __name__ == '__main__':
    train()
#     predict()
    
    
    