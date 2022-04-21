# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:23:51 2022

@author: aksel
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import cv2 as cv

folder = '../data_hw5_ext/calibration'

K           = np.loadtxt(join(folder, 'K.txt'))
dc          = np.loadtxt(join(folder, 'dc.txt'))
std_int     = np.loadtxt(join(folder, 'std_int.txt'))
u_all       = np.load(join(folder, 'u_all.npy'))
image_size  = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
mean_errors = np.loadtxt(join(folder, 'mean_errors.txt'))


fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

mu = dc[:5]
std_dev = np.array([k1,k2,p1,p2,k3])

im_path = '../data_hw5_ext/calibration/image042.jpg'

I = cv.imread(im_path)

for i in range(10):
    params = np.random.normal(mu,std_dev)
    print(params)
    I_undistorted = cv.undistort(src=I,cameraMatrix=K,distCoeffs=params)
    I_undistorted = I_undistorted[750:2000,200:2100]
    plt.imshow(I_undistorted)
    plt.show()
    