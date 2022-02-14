from textwrap import indent
import numpy as np
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
v = 0.01
receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)
recepsurface_ratio = 10
rate = 1
diffusion = 2 #ideally 0.1 #1 initially
seeds = np.arange(1,101)
distances = np.arange(2,12)
mean_final_counts = []
std_final_counts = []


for init_distance in distances:
    final_counts = []
    final_steps = []
    final_time =[]
    for seed in seeds: 
        print(seed, init_distance)
        cutoff = 30 #20 initially
        init_pos = np.matmul(special_ortho_group.rvs(3,1,random_state= seed),np.array([init_distance,0,0]))
        sourcex= init_pos[0]
        sourcey= init_pos[1]
        sourcez= init_pos[2]
        max_particles = 100000
        #print('# movement simulation of the cell with greedy algorithm')
        #print('# init_distance = '+str(init_distance))
        #print('# rate = '+str(rate))
        #print('# diffusion = '+str(diffusion))
        #print('# seed = '+str(seed))
        #print('# cutoff = '+str(cutoff))
        #print('# init_pos = '+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))
        #print('# max_particles = '+str(max_particles))

        # initalize c setup
        brownian_pipe, received, source = mlbi.init_BrownianParticle(sourcex,sourcey,sourcez,rate,diffusion,radius,seed,cutoff)
        ind_list = []
        countparticle = 0
        steps = 0
        count = 1 #count how many particles have been detected so far
        moving = 0
        t = []
        while(count <= max_particles):
            theta_mol = received[0]
            phi_mol = received[1]
            t.append(received[2])
            if count==1: tm = t[0]
            else: tm = t[count-1]-t[count-2]
            ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
            if ind == -1: 
                if moving ==0: 
                    received,source = mlbi.update_BrownianParticle(brownian_pipe)
                else: 
                    received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[indm][0],direction_sphcoords[indm][1], step_radius=tm * v)
                    if str(source[0]) == 'F':
                        print('# Source found')
                        break
                    else:
                        x,y,z = spherical2cart_point(source[1],source[2])
                        sourcex = source[0]* x
                        sourcey = source[0]* y
                        sourcez = source[0]* z
                    #print(str(count)+'\t'+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))
                        steps+=1
            else: #same index for receptors and directions
                moving =1
                indm = ind[0][0]
                received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[indm][0],direction_sphcoords[indm][1], step_radius=tm * v)
                if str(source[0]) == 'F':
                    print('# Source found')
                    break
                else:
                    x,y,z = spherical2cart_point(source[1],source[2])
                    sourcex = source[0]* x
                    sourcey = source[0]* y
                    sourcez = source[0]* z
                    #print(str(count)+'\t'+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))
                    steps+=1
            count+=1
        final_counts.append(count)
        final_steps.append(steps)
        final_time.append(t[count-1])

    #mean_counts = np.mean(final_counts)
    #range_counts = np.std(final_counts)
    with open("greedy_algorithm_stepsmoved_diff2cutoff30_v.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_steps)+'\n')
    with open("greedy_algorithm_counts_diff2cutoff30_v.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_counts)+'\n')
    with open("greedy_algorithm_time_diff2cutoff30_v.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_time)+'\n')