from pointssphere import *
import matplotlib.pyplot as plt
from haversine import * 
from sphericaltransf import *
from receptorpatches import *
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from itertools import product

def init_Receptors(radius, receptornum,random_yn, seed=0):
# Distribution of receptors:
# This code was taken from GitHub: https://gist.github.com/dinob0t/9597525
# Uses reference: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    if random_yn==1: x,y,z = random_on_sphere_points(radius,receptornum,seed=0)
    else: x,y,z = regular_on_sphere_points(radius,receptornum)
    theta,phi = cart2spherical_array(x,y,z) 
    receptor_sphcoords = np.concatenate(([theta],[phi])).T
    receptor_cartcoords = np.concatenate(([x],[y],[z])).T
    activation_receptors = np.zeros((receptornum))
    return receptor_sphcoords,receptor_cartcoords, activation_receptors

def visualize_Receptors(receptor_cartcoords,radius, mindistance):  
# To visualize the receptors create circle patches of mindistance assuming this distance is small compared to R, so this arclength can be 
# approximately equal to the radius of a flat circle patch for the purpose of visuals.
# If we want to make an exact visual we need to do projections of circles onto the sphere surface, which I have not been able to do yet.
# The functions rotiation matrix, patchpath_2d_to_3d and pathtranslate are all from:
# https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
# We create circle patches and then rotate and translate so they represent areas of receptors on the surface of the cell.
# The receptor centres are represented with black dots.

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    
    ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0) #Plot cell surface
    #ax.scatter(receptor_cartcoords[:,0],receptor_cartcoords[:,1],receptor_cartcoords[:,2])
    
    for i in range(0,len(receptor_cartcoords)):
        normal=(receptor_cartcoords[i,0],receptor_cartcoords[i,1],receptor_cartcoords[i,2])
        p = Circle((0,0), mindistance, facecolor = 'r', alpha = .7, fill=True) #mindistance is small so arclength ~= radius of circle patch
        ax.add_patch(p)
        pathpatch_2d_to_3d(p, z = 0, normal = normal)
        pathpatch_translate(p, normal)
    plt.show()
    return plt

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

#receptor_sphcoords,receptor_cartcoords, activation_receptors=init_Receptors(1,20)
#plt=visualize_Receptors(receptor_cartcoords,1,0.05)
#plt.show()
#receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(10,1,1)
#indx=activation_Receptors(0.1,0.4,receptor_sphcoords, 1, 2)
#print(indx)

