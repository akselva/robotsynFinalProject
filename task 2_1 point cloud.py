# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:14:18 2022

@author: aksel
"""
import numpy as np
import cv2 as cv
from triangulate_many import *
from figures import *
from decompose_E import *

K = np.loadtxt('../data_hw5_ext/calibration/K.txt')
kp1 = np.loadtxt('../data_hw5_ext/calibration/kp1.txt')
kp2 = np.loadtxt('../data_hw5_ext/calibration/kp2.txt')
kp1 = kp1.T
kp2 = kp2.T
kp1 = np.vstack([kp1,np.ones((1,kp1.shape[1]))])
kp2 = np.vstack([kp2,np.ones((1,kp2.shape[1]))])
E = np.loadtxt('../data_hw5_ext/calibration/E.txt')
inliers = np.loadtxt('../data_hw5_ext/calibration/inliers.txt').astype(bool)
I1 = cv.imread('../data_hw5_ext/IMG_8210.jpg')

uv1 = kp1[:,inliers]
uv2 = kp2[:,inliers]

T4 = decompose_E(E)
xy1 = np.linalg.solve(K,uv1)
xy2 = np.linalg.solve(K,uv2)

T4 = decompose_E(E)
best_num_visible = 0
for i, T in enumerate(T4):
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = T[:3,:]
    X1 = triangulate_many(xy1, xy2, P1, P2)
    X2 = T@X1
    num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
    if num_visible > best_num_visible:
        best_num_visible = num_visible
        best_T = T
        best_X1 = X1
T = best_T
X = best_X1

draw_point_cloud(X, I1,uv1, xlim=(-8,7), ylim=(-5,+5), zlim=(0,15))