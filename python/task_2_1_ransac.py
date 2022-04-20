# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:41:53 2022

@author: aksel
"""

import matplotlib.pyplot as plt
import numpy as np
from estimate_E_ransac import *

K = np.loadtxt('../data_hw5_ext/calibration/K.txt')
kp1 = np.loadtxt('../data_hw5_ext/calibration/kp1.txt')
kp2 = np.loadtxt('../data_hw5_ext/calibration/kp2.txt')

kp1 = kp1.T
kp2 = kp2.T

kp1 = np.vstack([kp1,np.ones((1,kp1.shape[1]))])
kp2 = np.vstack([kp2,np.ones((1,kp2.shape[1]))])

E,inliers = estimate_E_ransac(kp1,kp2,K,4,100)

np.savetxt('../data_hw5_ext/calibration/E.txt',E)
np.savetxt('../data_hw5_ext/calibration/inliers.txt',inliers)