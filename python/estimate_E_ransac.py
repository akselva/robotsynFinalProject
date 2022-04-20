import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *

def estimate_E_ransac(uv1, uv2, K, distance_threshold, num_trials):
    xy1 = np.linalg.solve(K,uv1)
    xy2 = np.linalg.solve(K,uv2)
    largest_set = np.array([])
    best_E = np.empty((3,3))
    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E = estimate_E(xy1[:,sample], xy2[:,sample])

        e_abs = np.abs(epipolar_distance(F_from_E(E,K),uv1,uv2))
        inlier_indexes = np.where(e_abs<=distance_threshold)[0]
        if inlier_indexes.size > largest_set.size:
            largest_set = inlier_indexes
            best_E = E
    
        print(i)

    E = estimate_E(xy1[:,largest_set],xy2[:,largest_set])
    return E,largest_set
