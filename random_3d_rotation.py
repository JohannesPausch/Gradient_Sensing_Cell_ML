 # random_3d_rotation : randomly change orientation of 3-dimensional data
from sphericaltransf import cart2spherical_array, cart2spherical_point
from scipy.stats import special_ortho_group
import numpy as np

# takes azimuth theta (0,pi) and polar angle phi (0,2pi), turns them randomly and returns their turned version. 
# numpy array of phi and theta can be used but must have the same shape
# The used distribution on the space of rotations is the Haar measure, i.e. it is uniform on SO(3)
# if the seed can be specified using use_seed. If not specfied, seeds will change automatically
def random_3d_rotation(theta,phi, use_seed=None):   
    try:
        np.random.seed(seed=use_seed)
        y=np.matmul(special_ortho_group.rvs(3),np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))#special_ortho_group.rvs(3)
        theta,phi =cart2spherical_point(y[0],y[1],y[2]) #transform to theta phi again
        return theta, phi
    except ValueError:
        if phi.shape != theta.shape:  print("Error in random_3d_rotation: Shapes of phi and theta don't match")
        else: print("Unknown error in random_3d_rotation")
        return theta,phi


