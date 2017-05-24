# @Time    : 2017/5/24 10:50
# @Author  : Aries
# @Site    : 
# @File    : main.py
# @Software: PyCharm Community Edition

import numpy as np
import Particle_Filter_with_demo.Partical_algorithms as PA

import cv2

def Get_Frame(video_file_path):
    """
    获取视频文件中所有的帧
    :param video_file_path:
    :return:
    """
    frames = []
    camera = cv2.VideoCapture(video_file_path)
    while True:
        res, frame = camera.read()
        if not res:
            break
        frames.append(frame)
    camera.release()
    return frames

if __name__ == "__main__":
    particle_num = 1000 # 1000个粒子
    Xstd_rgb = 50
    Xstd_pos = 25
    Xstd_vec = 5
    Xrgb_trgt = np.array([255, 0, 0])
    F_update = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

    frames = Get_Frame('../Person.wmv')
    frame_width, frame_height = frames[0].shape[0:2]
    print('**************帧宽和帧高********************')
    print(frame_width, frame_height)
    print('**************初始化粒子群参数*****************')
    X = PA.create_particle(frame_width, frame_height, particle_num)
    print(X)
    print('**************开始迭代计算*****************')
    for k in range(len(frames)):
        X = PA.update_particles(F_update, Xstd_pos, Xstd_vec, X)
        # print(X)
    print('**************开始进行极大似然估计运算*****************')
        