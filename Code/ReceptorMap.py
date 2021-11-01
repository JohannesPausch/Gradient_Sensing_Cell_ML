from pointssphere import *
import matplotlib.pyplot as plt
from haversine import * 
from sphericaltransf import *
#from receptorpatches import *
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from itertools import product
import math

def init_Receptors(radius, receptornum,random_yn, seed=0):
# Distribution of receptors:
# This code was taken from GitHub: https://gist.github.com/dinob0t/9597525
# Uses reference: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    if random_yn==1: x,y,z = random_on_sphere_points(radius,receptornum,seed=0)
    else: x,y,z = regular_on_sphere_points(radius,receptornum-1)
    theta,phi = cart2spherical_array(x,y,z) 
    receptor_sphcoords = np.concatenate(([theta],[phi])).T
    receptor_cartcoords = np.concatenate(([x],[y],[z])).T
    activation_receptors = np.zeros((len(receptor_sphcoords))) #not the same as receptnum sometimes
    return receptor_sphcoords,receptor_cartcoords, activation_receptors

def activation_Receptors(mol_theta,mol_phi,receptor_sphcoords, radius, mindistance):
    #Check wether a molecule is inside the area of a receptor and return the index of the receptor.
    #If not in the area return a negative number: no receptor is activated.
    receptornum=len(receptor_sphcoords)
    theta_molecule = np.full((receptornum,1), mol_theta)
    phi_molecule = np.full((receptornum,1),mol_phi)
    distance = haversine(radius,theta_molecule,phi_molecule,receptor_sphcoords[:,0].reshape(receptornum,1),receptor_sphcoords[:,1].reshape(receptornum,1))
    if np.amin(distance)<= mindistance:
        index_recept = np.where(distance == np.amin(distance))
        return index_recept
    else: return -1
#######################

