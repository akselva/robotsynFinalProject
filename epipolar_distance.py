import numpy as np

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    e = np.zeros(n)
    for i in range(n):
        e1_num = uv2[:,i].T@F@uv1[:,i]
        e2_num = uv1[:,i].T@(F.T)@uv2[:,i]
        e1_den = np.sqrt(((F@uv1[:,i])[0])**2 + ((F@uv1[:,i])[1])**2)
        e2_den = np.sqrt(((F@uv2[:,i])[0])**2 + ((F@uv2[:,i])[1])**2)
        e1 = e1_num/e1_den
        e2 = e2_num/e2_den
        e_avg = (e1 + e2)/2
        e[i] = e_avg
    return e
