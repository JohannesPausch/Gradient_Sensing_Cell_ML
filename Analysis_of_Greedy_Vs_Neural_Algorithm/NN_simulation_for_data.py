from textwrap import indent
import numpy as np
import os

import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'Code'))

import ML_Brownian_Interface as mlbi
import random_3d_rotation as r3dr
from scipy.stats import special_ortho_group
from ReceptorNeuralNetwork import *
from IdealDirection import *
from ReceptorMap import *
from datawriteread import *
from sphericaltransf import *

# initialise the cell (load directions, load mlp, load receptors)
# choose initial distance
receptornum = 10
direction_sphcoords = pick_direction(0,10)
radius = 1
recepsurface_ratio = 10

receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)
# ones that dont work dist = 13, seeds 52, 90. seed 8 and 17 for dist 15
filename = 'Total_mlp2'
rate = 1
diffusion = 1 #ideally 0.1
seeds = [102]
distances = [13]
mean_final_counts = []
std_final_counts = []

for init_distance in distances:
    final_counts =[]
    for seed in seeds: 
        print(seed, init_distance)
        cutoff = 20
        init_pos = np.matmul(special_ortho_group.rvs(3,1,random_state= seed),np.array([init_distance,0,0]))
        print(init_pos)
        sourcex= init_pos[0]
        sourcey= init_pos[1]
        sourcez= init_pos[2]
        particlenum = 100
        max_particles = 100000
        mlp = load_neural_network(filename)
        # print('# movement simulation of the cell with previously learnt neural network')
        # print('# init_distance = '+str(init_distance))
        # print('# rate = '+str(rate))
        # print('# diffusion = '+str(diffusion))
        # print('# seed = '+str(seed))
        # print('# cutoff = '+str(cutoff))
        # print('# init_pos = '+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))
        # print('# particlenum = '+str(particlenum))
        # print('# max_particles = '+str(max_particles))
        # print('# filename of neural network = '+filename)

        # initalize c setup
        brownian_pipe, received, source = mlbi.init_BrownianParticle(sourcex,sourcey,sourcez,rate,diffusion,radius,seed,cutoff)
        ind_list = []
        countparticle = 0
        count = 1 #count how many particles have been detected so far
        while(count <= max_particles):
            theta_mol = received[0]
            phi_mol = received[1]
            ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
            if ind == -1: 
                countparticle +=1
                ind_list.append(-1)
                if countparticle>particlenum:
                    ind_list.pop(0) 
            else:
                countparticle +=1
                ind_list.append(ind[0][0])
                if countparticle>particlenum:
                    ind_list.pop(0)
            if len(ind_list) == particlenum:
                activation_array = np.zeros((1,len(receptor_sphcoords)))
                for i in ind_list:
                    if i != -1:
                        activation_array[0,i]+=1
                move = mlp.predict(activation_array)
    
                if (move == 0).all():
                    received,source = mlbi.update_BrownianParticle(brownian_pipe)
                
                else:
                    move= list(move[0])
                    received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[move.index(1)][0],direction_sphcoords[move.index(1)][1], step_radius=0.1)
                  
                if str(source[0]) == 'F':
                    # print('# Source found')
                    print(count)
                    break
                else:
                    x,y,z = spherical2cart_point(source[1],source[2])
                    sourcex = source[0]* x
                    sourcey = source[0]* y
                    sourcez = source[0]* z
                    #print(str(count)+'\t'+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))

            else: pass
            count+=1
        final_counts.append(count)
    mean_counts = np.mean(final_counts)
    range_counts = np.std(final_counts)
    with open("neural_network_steps_data.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_counts)+'\n')
    mean_final_counts.append(mean_counts)
    std_final_counts.append(range_counts)
print('mean', mean_final_counts)
print('st dev', std_final_counts)







