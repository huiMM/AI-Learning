'''
@author: xusheng
'''

import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from six.moves import xrange
import os

class Datasets(object):
    def __init__(self, path, users_dim=671, movies_dim=9125, movies_sample_dim=50):
        self._path = path
        self._rating_value = 5.
        
        self._users_dim = users_dim
        self._users_dict = self._users_dict()
        
        self._movies_dim = movies_dim
        self._movies_csv_name = 'movies.csv'
        self._movies_dict = self._movies_dict()
        
        self._movies_sample_dim = movies_sample_dim
        
        self._ratings_csv_name = 'ratings.csv'
        self._rating_ar = self._load_ratings_csv()
        
    @property
    def users_dim(self):
        return self._users_dim
    
    @property
    def movies_sample_dim(self):
        return self._movies_sample_dim
    
    @property
    def movies_dim(self):
        return self._movies_dim
    
    @property
    def rating_ar(self):
        return self._rating_ar
    
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
        
        rating_ar = np.zeros((self._users_dim, self._movies_dim), dtype=np.float32)
        for i in xrange(size):
            # 0: userId, 1: movieId, 2: rating, 3: timestamp
            user_idx = self._users_dict[data_frame.iat[i, 0]]
            movie_idx = self._movies_dict[data_frame.iat[i, 1]]
            rating = data_frame.iat[i, 2] / self._rating_value
            rating_ar[user_idx][movie_idx] = rating
        return rating_ar
        
class SVD(object):
    def __init__(self, A):
        self._A = A
    
    @property
    def A(self):
        return self._A
    
    @property
    def sigma(self):
        return self._sigma
    
    @property
    def norm_sigma(self):
        return self._norm_sigma
    
    @property
    def u(self):
        return self._u
        
    @property
    def vt(self):
        return self._vt
    
    def computeSVD(self):
        self._u, self._sigma, self._vt = svd(self._A, full_matrices=True, compute_uv=True)
        return self._u, self._sigma, self._vt

    def normalized_sigma(self):
        self._norm_sigma = self._sigma / np.sum(self._sigma)
        return self._norm_sigma
        
    def reduceDim(self, threadhold=0.85):
        reduced_dim = 0
        sigma_dim = len(self._norm_sigma)
        sum = 0.0

        for i in xrange(sigma_dim):
    #         print("step %d, sum %f" % (i, sum))
            if sum >= threadhold:
                break
            sum += self._norm_sigma[i]
            reduced_dim += 1
    
        return reduced_dim
    
    def shrinkDim(self, target_dim=2):
        return self._u[:, 0:target_dim], self._sigma[0:target_dim], self._vt[0:target_dim, :]

def sigma_2D_ar(sigma, m_dim, n_dim):
    size = len(sigma)
    dim = min(size, m_dim, n_dim)
    ar = np.zeros((m_dim, n_dim), dtype=np.float32)
    for i in xrange(dim):
        ar[i][i] = sigma[i]
    return ar

def demo():
    ds = Datasets(os.path.join('..', 'data', 'ml-latest-small'), movies_sample_dim=20)

    # origin dimension
    svd = SVD(ds.rating_ar)
    u_1, sigma_1, vt_1 = svd.computeSVD()
#     sigma_2d_ar_ori = sigma_2D_ar(sigma_1, ds.users_dim, ds.movies_dim)
    # recalculate ar from (M x M) * (M x N) * (N x N)
#     cal_1 = np.dot(np.dot(u_1, sigma_2d_ar_ori), vt_1)
#     delta_1 = svd.A - cal_1
#     print(np.sum(delta_1))
    
    # threadhold
#     svd.normalized_sigma()
#     reduced_dim = svd.reduceDim(threadhold=0.9)
#     print("Reduced / Full: %d / %d" % (reduced_dim, len(svd.norm_sigma)))

    # shrink dimension to r_dim
    r_dim = 2
    u_2, sigma_2, vt_2 = svd.shrinkDim(target_dim=r_dim)
#     sigma_2d_ar = sigma_2D_ar(sigma_2, r_dim, r_dim)
    # recalculate ar from (M x R) * (R x R) * (R x N)
#     cal_2 = np.dot(np.dot(u_2, sigma_2d_ar), vt_2)
#     delta_2 = svd.A - cal_2
#     print(np.sum(delta_2))
    
    # distribution
    plt.plot(u_2[:, 0], u_2[:, 1], 'r*')
    v_2 = np.transpose(vt_2)
    plt.plot(v_2[:, 0], v_2[:, 1], 'bo')
    plt.show()
    
#     fig = plt.figure()
#     axe = Axes3D(fig)
#     axe.plot_surface(v_2[:, 0], v_2[:, 1], v_2[:, 2], rstride=1, cstride=1, alpha=0.5)


if __name__ == '__main__':
    demo()
