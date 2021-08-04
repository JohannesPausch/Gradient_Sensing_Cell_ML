import numpy as np
from ReceptorMap import *
from ML_Brownian_Interface import *
import random_3d_rotation 

############## Parameters for receptor map ###################
receptornum = 10
recepsurface_ratio = 100 # mindistance to activate receptor is radius/recepsurface_ratio 
particlenum = 20

sourcenum = 10
diffusionnum = 11
distancenum = 10
maxdistance = 10
radiusnum = 10
maxradius = 1
ratenum = 10
maxrate = 1

source_theta = np.array(sourcenum)
source_phi = np.array(sourcenum)
diffusion_constants  = np.logspace(-6,0,diffusionnum)
distance_from_source = np.linspace(0,maxdistance,distancenum)
rate = np.linspace(0.1,maxrate,ratenum)
radius_sphere = np.linspace(0.1,maxradius,radiusnum)



### action #####
# need to link matlab to python, not sure yet how


########### LOOPs for data #################
receptor_seed=1
receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(receptornum,1,receptor_seed)

for s in range(0,sourcenum):
    source_theta[s],source_phi[s] = random_3d_rotation(cart2spherical_point(0,0,1),s)
    sourcex,sourcey,sourcez = spherical2cart_point(source_theta[s],source_phi[s])
    #function to relate source coordinates to action direction -> make Y vector
    for r in radius_sphere:
        for distance in distance_from_source:
            for ra in rate:
                for dif in diffusion_constants:
                    for p in range(0,particlenum):
                        #needs source position and radius to be included in parameters
                        brownian_pipe,received = init_BrownianParticle(sourcex,sourcey,sourcez,r,distance,ra,dif,p ) 
                        #do we fix parameters:training,cutoff,events,iterations? most realistic are big numbers
                        theta_mol, phi_mol = cart2spherical_point(received[0],received[1],received[2])
                        ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius_sphere[r],mindistance)
                        if ind == -1: pass
                        else: activation_array[ind] += 1
                        received = update_BrownianParticle(brownian_pipe)# do we need this if it doesn't move?
                    #how to we organize the activation arrays?
                    #just a large matrix, all related to the same Y?

    