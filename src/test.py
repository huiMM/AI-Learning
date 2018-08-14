'''
@author: xusheng
'''

from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Reshape, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from PIL import Image
import argparse
import math
from six.moves import xrange


class Mnist(object):

    def __init__(self):
        pass
    
    def load_data(self):
        dataset = input_data.read_data_sets(os.path.join('..', 'data', 'mnist'), validation_size=0)
        return (dataset.train.images, dataset.train.labels), (dataset.test.images, dataset.test.labels) 

        
def generator_model():
    model = Sequential()
    model.add(Dense(units=1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, 5, padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=5, padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=5))
    model.add(Activation('tanh'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def train(BATCH_SIZE):
    mnist = Mnist()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # 获取mnist手写字体数据
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    print(X_train.shape[0])
    print(X_train.shape[1])
    print(X_train.shape[2])
    discriminator = discriminator_model()  # 初始化判别模型
    generator = generator_model()  # 初始化生成模型
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)  # 联合生成和判别模型
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)  # 编译生成和判别模型
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)  # 编译判别模型
    noise = np.zeros((BATCH_SIZE, 100))

    for epoch in xrange(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            image_batch = image_batch.transpose((0, 2, 3, 1))
            generated_images = generator.predict(noise, verbose=0)  # 生成图片
            if index % 20 == 0:
                generated_images_tosave = generated_images.transpose((0, 3, 1, 2))
                image = combine_images(generated_images_tosave)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(str(epoch) + "_" + str(index) + ".png")  # 图片命名
            print(image_batch.shape, generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)  # 计算判别模型loss
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False  # 　冻结判别模型参数
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)  # 　计算生成判别模型loss
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            # 保存模型
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE * 20, 100))
        for i in range(BATCH_SIZE * 20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) + (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        generated_images_tosave = generated_images.transpose((0, 3, 1, 2))
        image = combine_images(generated_images_tosave)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)    
    
