'''
@author: xusheng
'''

import tensorflow as tf
import pandas as pd
from six.moves import xrange
import os

class Datasets(object):
    def __init__(self, path):
        self._path = path
        
        self._movies_csv_name = 'movies.csv'
        self._movies_tfrecords_name = 'movies.tfrecords'
    
    def _check_movies_tfrecords_available(self):
        return os.path.exists(os.path.join(self._path, self._movies_tfrecords_name))

    def _parse_movies_to_tfexample(self, data, label):
        return tf.train.Example(features=tf.train.Features(feature={
            'data': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data)])), 
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(label, encoding='utf8')]))
        }))
    
    def save_movies_to_tfrecords(self):
        data_frame = pd.read_csv(filepath_or_buffer=os.path.join(self._path, self._movies_csv_name))
        size = data_frame.shape[0]
        
        writer = tf.python_io.TFRecordWriter(os.path.join(self._path, self._movies_tfrecords_name))
        try:
            print('movies dataset size: %d' % (size))
            for i in xrange(size):
                # 0: movieId, 1: title
                tf_example = self._parse_movies_to_tfexample(data_frame.iat[i, 0], data_frame.iat[i, 1])
                writer.write(tf_example.SerializeToString())
                if (i+1) % 1000 == 0:
                    print('- %d/%d, %2.2f%%' % ((i+1), size, (i+1)*100.0/size))
        finally:
            writer.close()
            print('- %d/%d, %2.2f%%' % ((i+1), size, (i+1)*100.0/size))
            print('movies dataset end')

    def _parse_tfrecord_movies(self, proto):
        features = {'data': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'label': tf.FixedLenFeature((), tf.string, default_value='')}
        parsed_features = tf.parse_single_example(proto, features)
        return parsed_features['data'], parsed_features['label']

    def load_movies_tfrecords(self):
        if not self._check_movies_tfrecords_available():
            self.save_movies_to_tfrecords()
        
        dataset = tf.data.TFRecordDataset(os.path.join(self._path, self._movies_tfrecords_name))
        dataset = dataset.map(self._parse_tfrecord_movies)
        return dataset.make_one_shot_iterator()

def tfrecords_demo():
    ds = Datasets(os.path.join('..', 'data', 'ml-latest-small'))
    ds_iter = ds.load_movies_tfrecords()
    
    data, label = ds_iter.get_next()
    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print('start loop')
        while True:
            try:
                x, y = sess.run([data, label])
                print("%s, %s" % (x, y))
            except tf.errors.OutOfRangeError:
                break
        print('end loop')

def main(_):
    tfrecords_demo()

if __name__ == '__main__':
    tf.app.run(main=main)
