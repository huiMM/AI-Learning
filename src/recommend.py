'''
@author: xusheng
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import xrange
import os
from numpy import dtype

class Datasets(object):
    def __init__(self, path):
        self._path = path
        self._rating_value = 5.
        
        self._users_dim = 671
        self._users_dict = self._users_dict()
        
        self._movies_dim = 9125
        self._movies_csv_name = 'movies.csv'
        self._movies_dict = self._movies_dict()
        
        self._ratings_csv_name = 'ratings.csv'
        self._ratings_np_ds = self._load_ratings_csv()
        
    def _users_dict(self):
        idx_np = np.arange(0, self._users_dim, dtype=np.int32)
        userid_np = np.arange(1, self._users_dim + 1, dtype=np.int32)
        return dict(zip(userid_np, idx_np))
    
    def _movies_dict(self):
        data_frame = pd.read_csv(filepath_or_buffer=os.path.join(self._path, self._movies_csv_name))
        # add idx column
        idx_np = np.arange(0, self._movies_dim, dtype=np.int32)
        movieid_np = np.array(data_frame['movieId'])
        return dict(zip(movieid_np, idx_np))

    def _load_ratings_csv(self):
        data_frame = pd.read_csv(filepath_or_buffer=os.path.join(self._path, self._ratings_csv_name))
        size = data_frame.shape[0]
        
        np_ds = np.zeros((self._users_dim, self._movies_dim), dtype=np.float32)
        for i in xrange(size):
            # 0: userId, 1: movieId, 2: rating, 3: timestamp
            user_idx = self._users_dict[data_frame.iat[i, 0]]
            movie_idx = self._movies_dict[data_frame.iat[i, 1]]
            rating = data_frame.iat[i, 2] / self._rating_value
            np_ds[user_idx][movie_idx] = rating
        return np_ds
        
class Recommend(object):
    def __init__(self):
        pass
    
    def computeSVD(self, tensor):
        self._sigma, self._u, self._v = tf.svd(tensor, full_matrices=True, compute_uv=True, name='svd')
        return self._sigma, self._u, self._v

    def reduceSigma(self):
        self._sigma = self._sigma / tf.reduce_sum(self._sigma, axis=None)
        return self._sigma
        
    def reduceDim(self, threadhold=0.85):
        reduced_dim = 0
        sigma_dim = len(self._sigma_val)
        sum = 0.0

        for i in xrange(sigma_dim):
    #         print("step %d, sum %f" % (i, sum))
            if sum >= threadhold:
                break
            sum += self._sigma_val[i]
            reduced_dim += 1
    
        return reduced_dim

def demo():
    ds = Datasets(os.path.join('..', 'data', 'ml-latest-small'))
    A_shape = (ds._users_dim, ds._movies_dim)
    A = tf.get_variable('A', shape=A_shape, dtype=tf.float32, initializer=tf.constant_initializer(ds._ratings_np_ds))
    
    r = Recommend()
    r.computeSVD(A)
    sigma_op = r.reduceSigma()

    init_op = tf.global_variables_initializer()
     
    with tf.Session() as sess:
        sess.run(init_op)
        r._sigma_val = sess.run(sigma_op)
    
    reduced_dim = r.reduceDim()
    print(r._sigma_val)
    print("Reduced / Full: %d / %d" % (reduced_dim, len(r._sigma_val)))

def main(_):
    demo()

if __name__ == '__main__':
    tf.app.run(main=main)
