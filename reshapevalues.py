import numpy as np

def reshapevalues(theta,phi):
    result = np.empty((2*theta.size), dtype=theta.dtype) 
    result[0::2]=theta
    result[1::2]=phi
    return result