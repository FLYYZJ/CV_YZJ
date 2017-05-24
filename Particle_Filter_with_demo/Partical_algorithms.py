# @Time    : 2017/5/24 15:35
# @Author  : Aries
# @Site    : 
# @File    : Partical_algorithms.py
# @Software: PyCharm Community Edition
import numpy as np

def create_particle(frame_width, frame_height, num_of_particle):
    """
    初始化粒子群
    :param frame_width: 帧宽
    :param frame_height: 帧高
    :param num_of_particle: 粒子数
    :return:
    """
    X1 = np.random.randint(1, frame_height, num_of_particle)  # 列向量
    X2 = np.random.randint(1, frame_width, num_of_particle)  # 列向量
    X3 = np.zeros((2, num_of_particle))  # 两行，num_of_particle列
    return np.vstack((X1, X2, X3))

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




if __name__ == '__main__':
    print(create_particle(20, 20, 10))
