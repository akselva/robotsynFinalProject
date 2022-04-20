import numpy as np

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.zeros((n,9))
    for i in range(n):
        A[i,0] = xy1[0,i]*xy2[0,i]
        A[i,1] = xy1[1,i]*xy2[0,i]
        A[i,2] = xy2[0,i]
        A[i,3] = xy1[0,i]*xy2[1,i]
        A[i,4] = xy1[1,i]*xy2[1,i]
        A[i,5] = xy2[1,i]
        A[i,6] = xy1[0,i]
        A[i,7] = xy1[1,i]
        A[i,8] = 1
    
    U,s,VT = np.linalg.svd(A)
    V = VT.T
    e = V[0:,-1]
    E = np.vstack([e[:3],e[3:6],e[6:]])
    
    return E
