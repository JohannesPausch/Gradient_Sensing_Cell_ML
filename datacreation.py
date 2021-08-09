from IdealDirection import *
import numpy as np
from ReceptorMap import *
from ML_Brownian_Interface import *
import random_3d_rotation 

def datacreate(
direction_sphcoords,
receptornum = 10,
recepsurface_ratio = 100,
particlenum = 20,
sourcenum = 10,
random_yn  = 0,
diffusionnum = 11,
diffusionexact = -1,
distancenum = 10,
maxdistance = 10,
distanceexact = -1,
radiusnum = 10,
maxradius = 1,
radiusexact = -1,
ratenum = 10,
maxrate = 1,
rateexact = -1, 
receptor_seed = 1): 
############## Parameters for receptor map ###################
#the variables ...exact are a shortcut to skipping all the for loops and plugging in one value per variable
#random_yn decides if we want to take a random uniform approach for the data (1) or if we want to do equally spaced values between 
#chosen boundaries (0).

    source_theta = np.array(sourcenum)
    source_phi = np.array(sourcenum)
    
    
    if diffusionexact== -1:
        if random_yn==0:
            diffusion_constants  = np.logspace(-6,0,diffusionnum)
        elif random_yn==1:
            diffusion_constants = np.random.default_rng().uniform(0,1, diffusionnum)
        else: 
            raise ValueError("Pick if diffusion constants should be equally spaced (random_yn = 0) or randomly chosen (random_yn = 1)")
    else: diffusion_constants = diffusionexact
    if distanceexact== -1:
        if random_yn==0:
            distance_from_source = np.linspace(0,maxdistance,distancenum)
        elif random_yn==1:
            distance_from_source = np.random.default_rng().uniform(0,maxdistance, distancenum)
        else:
            raise ValueError("Pick if distance constants should be equally spaced (random_yn = 0) or randomly chosen (random_yn = 1)")
    else: distance_from_source = distanceexact
    if rateexact == -1:
        if random_yn==0:
            rate = np.linspace(0.1,maxrate,ratenum)
        elif random_yn==1:
            rate = np.random.default_rng().uniform(0,maxrate, ratenum)
        else:
            raise ValueError("Pick if rate constants should be equally spaced (random_yn = 0) or randomly chosen (random_yn = 1)")
    else: rate=rateexact
    if radiusexact== -1:
        if random_yn==0:
            radius_sphere = np.linspace(0.1,maxradius,radiusnum)
        elif random_yn==1:
            radius_sphere = np.random.default_rng().uniform(0,maxradius, radiusnum)
        else:
            raise ValueError("Pick if radius constants should be equally spaced (random_yn = 0) or randomly chosen (random_yn = 1)")
    else: radius_sphere = radiusexact


########### LOOPs for data #################
    #fix number of receptors for each training data, it's like fixing the number of eyes the cell has... makes sense, I think.
    receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(receptornum,1,receptor_seed)
    #initialize activation 
    activation_matrix = np.zeros(receptornum) 
    #initialize X
    X = np.zeros((receptornum*sourcenum,len(radius_sphere)*len(distance_from_source)*len(rate)*len(diffusion_constants)*particlenum))
   
    for s in range(0,sourcenum-1):
        #pick source??
        #source_theta[s],source_phi[s] = random_3d_rotation(cart2spherical_point(0,0,1),s)
        #sourcex,sourcey,sourcez = spherical2cart_point(source_theta[s],source_phi[s])
        #function to relate source coordinates to action direction -> make Y vector
        #Y = ideal_direction(source_theta[s],source_phi[s],direction_sphcoords, 1)

        for r in radius_sphere:
            mindistance = radius_sphere/recepsurface_ratio
            for distance in distance_from_source:
                for ra in rate:
                    for dif in diffusion_constants:
                        activation_array = np.zeros(receptornum)
                            #needs source position and radius to be included in parameters?
                        brownian_pipe,received = init_BrownianParticle_test(r,distance,ra,dif,s ) 
                        Y = ideal_direction(received[3],received[4],direction_sphcoords, 1)
                            #same seed for brownian_pipe if we want to initialize with the same source rotation?
                            #do we fix parameters training,cutoff,events,iterations? 
                        count = 1 #count how many particles in one activation array measure. Starts with 1 particle.
                        while(count <= particlenum):
                            theta_mol = received[0]
                            phi_mol = received[1]
                            ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,r,mindistance)
                            if ind == -1: pass
                            else: activation_array[ind] += 1
                            
                            if activation_matrix==np.zeros(receptornum): activation_matrix = activation_array
                            else: 
                                activation_matrix = np.concatenate(([activation_matrix],[activation_array]))
                            Y = np.concatenate(([Y],[Y])).T

                            received = update_BrownianParticle_test(brownian_pipe)
                            count+=1

        X[s*len(activation_matrix),:] = activation_matrix #rows are each activation array
        Y[:,s*len(activation_matrix)] = Y # columns are source direction, corresponding to each activation array row
    return X, Y
