import numpy as np
from pointssphere import *
from sphericaltransf import *
from haversine import * 
import random
import math
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

def random_directions(number):
    # Cart coords not necessary just theta and phi
    x,y,z = random_on_sphere_points(1, number-1) 
    theta,phi= cart2spherical_receptors(x,y,z) 
    direction_sphcoords= np.array([theta,phi]).T

    return direction_sphcoords
    
    
def regular_directions(number):
    # Cart coords not necessary just theta and phi
    x,y,z = regular_on_sphere_points(1, number-1)
    theta,phi=  cart2spherical_receptors(x,y,z) 
    direction_sphcoords= np.array([theta,phi]).T

    return direction_sphcoords
     

def fibionaaci_directions(number):
    # found this function on stackoverflow https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    # thought it might be a good way to distribute the directions on a sphere
    indices = np.arange(0, number, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/number)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    direction_sphcoords= np.array([theta,phi]).T
    
    return direction_sphcoords
    


def ideal_direction(source_theta, source_phi, direction_sphcoords, radius):
    
    directionnum=len(direction_sphcoords)
    theta_source = np.full((directionnum,1), source_theta)
    phi_source = np.full((directionnum,1), source_phi)
    distance = haversine(radius,theta_source,phi_source,direction_sphcoords[:,0].reshape(directionnum,1), direction_sphcoords[:,1].reshape(directionnum,1))

    idx = np.where(distance == np.amin(distance))
    best_direction = direction_sphcoords[idx[0],:]
    Y = np.zeros(len(direction_sphcoords))
    Y[idx[0]] = 1
  
    return Y
    

if __name__ == '__main__':
    
    
    source_theta = 0.89
    source_phi = 1.5
    radius = 1
    step = 0.1
    direction_sphcoords, _ = regular_directions(1, 8)

    direction_cell_moves = ideal_direction(source_theta, source_phi, direction_sphcoords, radius)
    print(direction_cell_moves)
    
    
