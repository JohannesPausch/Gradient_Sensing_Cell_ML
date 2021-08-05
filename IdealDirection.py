import numpy as np
from pointssphere import *
from sphericaltransf import *
from haversine import * 
import random
import math
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import matlab.engine


def random_directions(radius, number):
    # this function uses Paula's receptor code for directions on a sphere
    x,y,z = random_on_sphere_points(radius, number-1) 
    theta,phi= cart2spherical_receptors(x,y,z) 
    direction_sphcoords= np.array([theta,phi]).T
    direction_cartcoords= np.array([x,y,z]).T

    return direction_sphcoords, direction_cartcoords
    
    
def regular_directions(radius, number):
    # this function uses Paula's receptor code for regularly spaced directions on a sphere
    x,y,z = regular_on_sphere_points(radius, number-1)
    theta,phi=  cart2spherical_receptors(x,y,z) 
    direction_sphcoords= np.array([theta,phi]).T
    direction_cartcoords= np.array([x,y,z]).T

    return direction_sphcoords, direction_cartcoords
     

def fibionaaci_directions(radius, number):
    # found this function on stackoverflow https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    # thought it might be a good way to distribute the directions on a sphere
    indices = np.arange(0, number, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/number)
    theta = np.pi * (1 + 5**0.5) * indices
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    direction_sphcoords= np.array([theta,phi]).T
    direction_cartcoords= np.array([x,y,z]).T
    
    return direction_sphcoords, direction_cartcoords
     
    
def eq_spaced_directions(radius, number):
 # I don't know how to run this matlab function sorry
    eng = matlab.engine.start_matlab()
    eng.eq_point_set_polar(2,3)
    


def ideal_direction(source_theta, source_phi, direction_sphcoords, radius):
    
    receptornum=len(direction_sphcoords)
    theta_source = np.full((receptornum,1), source_theta)
    phi_source = np.full((receptornum,1), source_phi)
    distance = haversine(radius,theta_source,phi_source,direction_sphcoords[:,0].reshape(receptornum,1), direction_sphcoords[:,1].reshape(receptornum,1))

    idx = np.where(distance == np.amin(distance))
    best_direction = direction_sphcoords[idx[0],:]
    sph_vector_move = np.array([radius, best_direction]).T #this is the vector from the origin that the source should move in in r, theta, phi. 
    
    return sph_vector_move
    

if __name__ == '__main__':
    
    
    source_theta = 
    source_phi = 
    direction_sphcoords, _ = regular_directions(1, 8)

    direction_cell_moves = ideal_direction(source_theta, source_phi, direction_sphcoords, 1)
    
    
    
