# @Time    : 2017/5/18 12:39
# @Author  : Yao zijie
# @Site    : 
# @File    : main.py
# @Software: PyCharm Community Edition
# @Interpreter: Python 3.5.1

import cv2
import Cameshift_with_demo.Feature_caculation as FC
import Cameshift_with_demo.Meanshift as meanshift
import Cameshift_with_demo.GetRoi as GetRoi


# 获取背景图
# background = pickle.load(open('background.pkl', 'rb'))
#
frames_hsv, frames_gray, frames_color = GetRoi.GetFrame('../Ball.avi')
# 第1个目标的位置以及特征
frame_hist_1, q_max_1, ix_1, iy_1, w_1, h_1 = FC.Hist_cal(frames_hsv[0], 197, 56, 17, 14)
# 第2个目标的位置以及特征
frame_hist_2, q_max_2, ix_2, iy_2, w_2, h_2 = FC.Hist_cal(frames_hsv[0], 193, 78, 16, 13)
# 第3个目标的位置以及特征
frame_hist_3, q_max_3, ix_3, iy_3, w_3, h_3 = FC.Hist_cal(frames_hsv[0], 190, 96, 14, 13)


for i in range(1, len(frames_hsv)):
    back_projection_img_1, ix_1, iy_1, w_1, h_1 = meanshift.MeanShift(frames_hsv[i], frame_hist_1, q_max_1, ix_1, iy_1,
                                                                       w_1, h_1, 1, 3)
    back_projection_img_2, ix_2, iy_2, w_2, h_2 = meanshift.MeanShift(frames_hsv[i], frame_hist_2, q_max_2, ix_2, iy_2,
                                                                      w_2, h_2, 2, 2)
    back_projection_img_3, ix_3, iy_3, w_3, h_3 = meanshift.MeanShift(frames_hsv[i], frame_hist_3, q_max_3, ix_3, iy_3,
                                                                       w_3, h_3, 3, 3)

    try:
        cv2.rectangle(frames_color[i], (iy_1, ix_1), (iy_1 + h_1, ix_1 + w_1), (0, 255, 0), 2)
        cv2.rectangle(frames_color[i], (iy_2, ix_2), (iy_2 + h_2, ix_2 + w_2), (255, 0, 0), 2)
        cv2.rectangle(frames_color[i], (iy_3, ix_3), (iy_3 + h_3, ix_3 + w_3), (0, 0, 255), 2)
        cv2.imshow('img', frames_color[i])
        cv2.imshow('back_projection_img1', back_projection_img_1)
        cv2.imshow('back_projection_img2', back_projection_img_2)
        cv2.imshow('back_projection_img3', back_projection_img_3)
        if cv2.waitKey(110) & 0xff == 27:
            break
        # cv2.waitKey(0)
    except:
        cv2.imshow('back_projection_img1', back_projection_img_1)
        cv2.imshow('back_projection_img2', back_projection_img_2)
        cv2.imshow('back_projection_img2', back_projection_img_3)
        cv2.waitKey(0)

