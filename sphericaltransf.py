import numpy as np
import math

def cart2spherical_point(x,y,z):
    r = math.pow(x,2) + math.pow(y,2) + math.pow(z,2)
    theta = math.pi/2 - math.acos(z/r)
    phi = math.atan2(y,x)
    return r,theta,phi

def cart2spherical_receptors(x,y,z):
    receptnum = len(x)
    r = np.zeros((receptnum))
    theta = np.zeros((receptnum))
    phi = np.zeros((receptnum))
    for i in range(0,receptnum):
        r[i], theta[i], phi[i] = cart2spherical_point(x[i],y[i],z[i]) 
    return theta,phi

def spherical2cart_point(theta,phi):
    x = math.sin(theta)*math.cos(phi)
    y = math.sin(theta)*math.sin(phi)
    z = math.cos(theta)
    return x,y,z

