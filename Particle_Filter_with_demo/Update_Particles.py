# @Time    : 2017/5/24 11:18
# @Author  : Aries
# @Site    : 
# @File    : Update_Particles.py
# @Software: PyCharm Community Edition

import numpy as np

def update_particles(F_update, Xstd_pos, Xstd_vec, X):
    """

    :param F_update:
    :param Xstd_pos:
    :param Xstd_vec:
    :param X:
    :return:
    """
    X = np.dot(F_update, X)
    X[0:2, :] = X[0:2, :] + Xstd_pos * np.random.randint(2, X.shape[1])
    X[2:4, :] = X[2:4, :] + Xstd_vec * np.random.randint(2, X.shape[1])
    return X