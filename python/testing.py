# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:10:13 2022

@author: aksel
"""

import numpy as np
from figures import *
from F_from_E import *
import cv2 as cv

K = np.loadtxt('../data_hw5_ext/calibration/K.txt')
kp1 = np.loadtxt('../data_hw5_ext/calibration/kp1.txt')
kp2 = np.loadtxt('../data_hw5_ext/calibration/kp2.txt')
kp1 = kp1.T
kp2 = kp2.T
kp1 = np.vstack([kp1,np.ones((1,kp1.shape[1]))])
kp2 = np.vstack([kp2,np.ones((1,kp2.shape[1]))])

E = np.loadtxt('../data_hw5_ext/calibration/E.txt')
inliers = np.loadtxt('../data_hw5_ext/calibration/inliers.txt').astype(int)

F = F_from_E(E,K)

I1 = cv.imread('../data_hw5_ext/IMG_8220.jpg')
I2 = cv.imread('../data_hw5_ext/IMG_8221.jpg')


draw_correspondences(I1,I2,kp1,kp2,F)