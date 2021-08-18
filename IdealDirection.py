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
    theta,phi= cart2spherical_array(x,y,z) 
    direction_sphcoords= np.array([theta,phi]).T

    return direction_sphcoords
    
    
def regular_directions(number):
    # Cart coords not necessary just theta and phi
    x,y,z = regular_on_sphere_points(1, number-1)
    theta,phi=  cart2spherical_array(x,y,z) 
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
    
    source_phi = source_phi - np.pi
    directionnum=len(direction_sphcoords)
    theta_source = np.full((directionnum,1), source_theta)
    phi_source = np.full((directionnum,1), source_phi)
    distance = haversine(radius,theta_source,phi_source,direction_sphcoords[:,0].reshape(directionnum,1), direction_sphcoords[:,1].reshape(directionnum,1))
    #can detect distances that are the same and so have Y with multiple 1s, keep?
    idx = np.where(distance == np.amin(distance))
    distance = []
    for d in direction_sphcoords:
        distance.append(haversine(radius,source_theta,source_phi, d[0].round(8), d[1].round(8)))    
    Y = np.zeros((1,len(direction_sphcoords)))
    idx = np.where(distance == np.amin(distance))
    Y[0,idx[0]] = 1
    return Y

def pick_direction(m = 0,num = 20):
    if m == 0:
        return regular_directions(num)
    elif m == 1:
        return fibionaaci_directions(num)
    elif m == 2:
        return random_directions(num)
    else:
        raise ValueError("Method is not valid. Pick method to create directions of cell, m = 0 regular directions, m = 1 fibonacci , m = 2 random directions")


