import numpy as np
import math

def cart2spherical_point(x,y,z):
    r = math.pow(x,2) + math.pow(y,2) + math.pow(z,2)
    theta = math.pi/2 - math.acos(z/r)
    if (z!=0.):
        theta=math.atan(np.sqrt(x*x + y*y)/z)
        if (theta<0.): theta+=math.pi
    else:theta=math.pi/2.
    phi = math.atan2(y,x)
    if y < 0 : phi += 2*np.pi
    return theta,phi

def cart2spherical_array(x,y,z):
    length = len(x)
    r = np.zeros((length))
    theta = np.zeros((length))
    phi = np.zeros((length))
    for i in range(0,length):
        theta[i], phi[i] = cart2spherical_point(x[i],y[i],z[i]) 
    return theta,phi

def spherical2cart_point(theta,phi):
    x = math.sin(theta)*math.cos(phi)
    y = math.sin(theta)*math.sin(phi)
    z = math.cos(theta)
    return x,y,z
