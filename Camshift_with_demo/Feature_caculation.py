# @Time    : 2017/5/18 12:37
# @Author  : Yao zijie
# @Site    : 
# @File    : Feature_caculation.py
# @Software: PyCharm Community Edition
# @Interpreter: Python 3.5.1

import cv2
import numpy as np
try:
    import Cameshift_with_demo.GetRoi as GetRoi
except:
    import GetRoi as GetRoi


def HSV_frame_H_Cal(frame):
    """
    HSV_frame_H_Cal函数 提取通道H
    :param frame: 输入一帧图像，该帧图像是HSV图像
    :return:
    """
    return cv2.split(frame)[0]

def Hist_cal(frame, ix, iy, w, h):
    """
    Hist_cal 计算目标区域的直方图特征，仅提取其中的H通道进行直方图计算
    :param frame:
    :param ix: 初始找到的目标位置，左上角x坐标
    :param iy: 初始找到的目标位置，左上角y坐标
    :param w:  初始找到的目标的宽度
    :param h:  初始找到的目标的高度
    :return:
    """
    H = HSV_frame_H_Cal(frame)  # 提取H通道
    frame_hist = np.histogram(H[ix:ix+w, iy:iy+h], bins=12, range=[0, 180])   # 计算目标区域的H通道直方图特征

    return frame_hist, np.max(frame_hist[0]), ix, iy, w, h

def Back_projection_cal(frame_H, frame_hist, q_max):
    """
    Back_projection_cal 反投影计算函数，将图片像素反投影为特征图
    :param frame_H: 输入的一帧图像
    :param frame_hist: 目标区域的直方图特征
    :param q_max:  目标区域中直方图特征的最大值
    :return:
    """
    back_projection_img = np.zeros(frame_H.shape, dtype=np.uint8)
    for i in range(0, 180, 15):  # 利用numpy的特性高效计算反投影图，不用Python的for循环，提高效率
        back_projection_img[(frame_H >= i) & (frame_H < i + 15)] = frame_hist[0][(frame_hist[1] == i)[:12]] * 255 // q_max
    return back_projection_img


def Centroid_Mass_Cal(back_projection_img, ix, iy, w, h):
    """
    Centroid_Mass_Cal 目标质心，目标面积计算,
    :param back_projection_img: 输入一张特征图
    :param ix: 目标区域当前的左上角x坐标
    :param iy: 目标区域当前的右上角y坐标
    :param w: 目标区域当前的宽度
    :param h: 目标区域当前的高度
    :return:
    """
    M00 = np.sum(back_projection_img[ix:ix+w, iy:iy+h])  # 计算一阶矩
    xx = np.array([[x] for x in range(ix, ix + w)])
    M10 = np.sum(xx * back_projection_img[ix:ix + w, iy:iy + h])  # 利用numpy的广播操作计算二阶矩，提高效率
    yy = np.array([y for y in range(iy, iy + h)])
    M01 = np.sum(yy * back_projection_img[ix:ix + w, iy:iy + h])  # 利用numpy的广播操作计算二阶矩，提高效率
    xc = M10 // M00  # 新目标中心的x坐标
    yc = M01 // M00  # 新目标中心的y坐标
    S = 2 * np.sqrt(M00)  # 获取目标的面积大小

    return xc, yc, S


if __name__ == '__main__':
    frames, frames_gray, frames_color = GetRoi.GetFrame('../Ball.avi')
    H = HSV_frame_H_Cal(frames[10])
    frame_hist, q_max, ix, iy, w, h = Hist_cal(frames[0])
    back_projection_img = Back_projection_cal(H, frame_hist, q_max)
    cv2.imshow('int', frames_color[1][ix:ix + w, iy:iy + h])
    cv2.imshow('gg', frames[10])
    cv2.imshow('initint', back_projection_img)
    cv2.waitKey(0)

    print(Centroid_Mass_Cal(back_projection_img, ix, iy, w, h))
    pass





