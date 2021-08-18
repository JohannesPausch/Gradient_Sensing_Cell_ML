import numpy as np
import math
#help from: https://janakiev.com/blog/gps-points-distance-python/
#Calculates the min distance on surface of sphere between two points of different theta,phi coords.
def haversine(R,theta1,phi1,theta2,phi2):
    dphi      = phi2 - phi1 #longitude conversion not necessary fifo.py does -180 to 180
    dtheta    = (math.pi/2-theta2) - (math.pi/2-theta1) #latitude conversion since should be -90 to 90 but theta is 0 to 180, nothing changes
    a = np.power(np.sin(dtheta/2),2) + np.cos(theta1)*np.cos(theta2)*np.power(np.sin(dphi/2),2)
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

