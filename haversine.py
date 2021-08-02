import numpy as np
import math
#help from: https://janakiev.com/blog/gps-points-distance-python/
#Calculates the min distance on surface of sphere between two points of different theta,phi coords.
def haversine(R,theta1,phi1,theta2,phi2):
    dphi      = phi2 - phi1
    dtheta    = theta2 - theta1
    a = np.power(np.sin(dtheta/2),2) + np.cos(theta1)*np.cos(theta2)*np.power(np.sin(dphi/2),2)
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

