# @Time    : 2017/5/18 12:39
# @Author  : Yao zijie
# @Site    : 
# @File    : Meanshift.py
# @Software: PyCharm Community Edition
# @Interpreter: Python 3.5.1

import cv2
import numpy as np
import Cameshift_with_demo.Feature_caculation as FC

DISTANCE_THRESHOLD = 1  # 距离差距，如果移动距离小于1，则表明到达目标中心
ITERATION_TIMES = 30  # meanshift算法的迭代轮数，最多进行30轮迭代

def MeanShift(frame,frame_hist, q_max, ix, iy, w, h,erode_1, erode_2):
    """
    MeanShift 算法实现，核心函数
    :param frame: 输入的一帧图片，这帧图片已经被转化为HSV图片
    :param frame_hist: 对应的目标区域的统计直方图
    :param q_max: 目标区域中统计直方图中的最大值
    :param ix: 当前目标的位置，左上角点x坐标
    :param iy: 当前目标的位置，右上角点y坐标
    :param w: 目标的宽度，相对x坐标
    :param h: 目标的高度，相对y坐标
    :param erode_1: 腐蚀算子大小的选择
    :param erode_2: 腐蚀算子大小的选择
    :return:
    """
    # 转HSV，并取H通道
    frame_H = FC.HSV_frame_H_Cal(frame)
    # 计算反投影图，并做腐蚀膨胀并阈值化
    back_projection_img = FC.Back_projection_cal(frame_H, frame_hist, q_max)
    back_projection_img = cv2.threshold(back_projection_img, 230, 255, cv2.THRESH_BINARY)[1]
    back_projection_img = cv2.erode(back_projection_img,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_1, erode_2)),
                                    iterations=2)
    back_projection_img = cv2.dilate(back_projection_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                                     iterations=2)
    # 保留之前算出的中心位置
    xc_ = int(ix + w // 2)
    yc_ = int(iy + h // 2)
    w_ = w
    h_ = h
    # 迭代寻找最优中心点
    for j in range(ITERATION_TIMES):
        while(True):
            try:
                xc, yc, S = FC.Centroid_Mass_Cal(back_projection_img, ix, iy, w, h)  # 计算出当前目标的中心位置，面积
                if xc > 0 and yc > 0:  # 一个防错策略，防止因目标移速过快，跳出了初始搜索框，因此判断是否在初始搜索框找到目标，如果没有就扩大搜索框
                    break
                else:  # 扩大搜索框
                    ix = ix -10
                    w = w + 40
                    h = h + 40
            except: # 另一个策略，如果当前无法找到目标，则保留原先的位置，等待下一轮进行搜索，这样可以减少中间丢失目标时程序出错跳出
                xc = xc_
                yc = yc_
                w = w_
                h = h_
                break
        if np.sqrt((ix + w / 2 - xc) ** 2 + (iy + h / 2 - yc) ** 2) < DISTANCE_THRESHOLD:  # 判断找到合适的匹配位置，是就退出
            break

        ix = int(xc - np.sqrt(S) // 2)  # 更新左上角x坐标
        iy = int(yc - np.sqrt(S) // 2)  # 更新左上角y坐标
        w = int(np.sqrt(S))  # 更新目标的宽度
        h = int(np.sqrt(S))  # 更新目标的高度

    return back_projection_img, ix, iy, w, h
