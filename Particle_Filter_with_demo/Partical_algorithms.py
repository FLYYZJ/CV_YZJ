# @Time    : 2017/5/24 15:35
# @Author  : Aries
# @Site    : 
# @File    : Partical_algorithms.py
# @Software: PyCharm Community Edition
import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_particle(frame_width, frame_height, num_of_particle):
    """
    初始化粒子群
    :param frame_width: 帧宽
    :param frame_height: 帧高
    :param num_of_particle: 粒子数
    :return:
    """
    # 下面4*4000的矩阵，每一列为一个粒子的状态
    X1 = np.random.randint(1, frame_height, num_of_particle)  # 列向量，代表粒子的x坐标
    X2 = np.random.randint(1, frame_width, num_of_particle)  # 列向量，代表粒子的y坐标
    X3 = np.zeros((2, num_of_particle))  # 两行，num_of_particle列,代表粒子的速度
    return np.vstack((X1, X2, X3))

def update_particles(F_update, Xstd_pos, Xstd_vec, X):
    """

    :param F_update:
    :param Xstd_pos:
    :param Xstd_vec:
    :param X:
    :return:
    """
    X = np.dot(F_update, X)  # 乘上一个F_update函数，此时粒子会发生运动
    X[0:2, :] = X[0:2, :] + Xstd_pos * np.random.randn(2, X.shape[1])  # 原位置 + 方差项 * 随机噪声，更新每个粒子的位置值
    X[2:4, :] = X[2:4, :] + Xstd_vec * np.random.randn(2, X.shape[1])  # 同理，速度也是原速度 + 方差项 * 随机噪声，更新每个粒子的速度
    return X

def calc_log_likelihood(Xstd_rgb, Xstd_trgt, X, Y):
    """
    计算极大似然函数，判断是否符合极大似然准则
    :param Xstd_rgb:
    :param Xstd_trgt:
    :param X:
    :param Y:
    :return:
    """
    Npix_h = Y.shape[0]
    Npix_w = Y.shape[1]
    N = X.shape[1]

    L = np.zeros((1, N))
    # Y = np.transpose(Y.reshape(480, 640, 3), (2, 0, 1))
    # Y1 = Y[1, 1, :]
    # Y2 = Y[:, 1, 1]
    A = -np.log(np.sqrt(2 * np.pi) * Xstd_rgb)
    B = -0.5 / (Xstd_rgb ** 2)

    X = np.round(X)

    for k in range(N):

        m = X[0, k]
        n = X[1, k]

        if (m >= 0 and m <= Npix_h - 1) and (n >= 0 and n <= Npix_w - 1):
            C = Y[int(m), int(n), :]
            D = C - Xstd_trgt
            D2 = np.dot(D.T, D)
            L[:, k] = A + B * D2
        else:
            L[:, k] = -np.inf
    return L

def resample_particles(X, L_log):
    """

    :param X:
    :param L_log:
    :return:
    """
    L = np.exp(L_log - np.max(L_log))
    Q = L / np.sum(L)

    R = np.cumsum(Q, 1)[0]

    N = X.shape[1]
    T = np.random.rand(N)

    I = np.digitize(T, R)

    X = X[:, I]
    return X

def show_particles(X, Y_k):
    for i in range(X.shape[1]):
        cv2.circle(Y_k, (int(X[1, i]), int(X[0, i])), 3, (0, 255, 0), -1)
    cv2.imshow('img', Y_k)
    cv2.waitKey(0)

if __name__ == '__main__':
    print(create_particle(20, 20, 10))
