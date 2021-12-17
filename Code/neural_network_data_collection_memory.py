from math import dist
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
recepsurface_ratio = 10

receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)
# ones that dont work dist = 13, seeds 52, 90. seed 8 and 17 for dist 15
filename = 'Total_mlp2'
rate = 1
diffusion = 2 #ideally 0.1
seeds = np.arange(1,100,1)
distances = np.arange(2,21,1)

mem = 2
#final_counts = []
#inal_steps = []
#final_time = []
#init_distance = 6
#for init_distance in distances:
for init_distance in distances:
    final_counts =[]
    final_steps = []
    final_time = []
    for seed in seeds: 
        #print(seed, init_distance)
        cutoff = 30
        init_pos = np.matmul(special_ortho_group.rvs(3,1,random_state= seed),np.array([init_distance,0,0]))
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
        steps=0
        memory = 0
        
        while(count <= max_particles):
            #print('hi2')
            theta_mol = received[0]
            phi_mol = received[1]
            tm = received[2]
            #print (theta_mol, phi_mol, received[2])
            #print(count)
            ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
            if ind == -1: 
                countparticle +=1
                ind_list.append(-1)
                if countparticle>particlenum:
                    #print(ind_list)
                    memory +=1 
                    ind_list.pop(0) 
            else:
                countparticle +=1
                ind_list.append(ind[0][0])

                if countparticle>particlenum:
                    #if ind in ind_list: del ind_list[ind_list.index(ind)]
                    #else: ind_list.pop(0)
                    #print(ind_list)
                    memory +=1 
                    ind_list.pop(0)
            if memory < mem :
                #print(memory)
                #if memory==1:
                #    print(ind_list)
                received,source = mlbi.update_BrownianParticle(brownian_pipe) 
                continue 
           # print(ind_list)
           # print('hi3')
            if len(ind_list) == particlenum:# and ind!=-1:
                activation_array = np.zeros((1,len(receptor_sphcoords)))
                for i in ind_list:
                    if i != -1:
                        activation_array[0,i]+=1
                
                move = mlp.predict(activation_array)
                #print(activation_array)
                memory = 0
                if (move == 0).all():
                    received,source = mlbi.update_BrownianParticle(brownian_pipe)
                
                else:
                    move= list(move[0])
                    received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[move.index(1)][0],direction_sphcoords[move.index(1)][1], step_radius=0.1)
                    steps+=1

                if str(source[0]) == 'F':
                    # print('# Source found')
                    break
                else:
                    x,y,z = spherical2cart_point(source[1],source[2])
                    sourcex = source[0]* x
                    sourcey = source[0]* y
                    sourcez = source[0]* z
                    #print(str(count)+'\t'+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))

            else:
                received,source = mlbi.update_BrownianParticle(brownian_pipe) 
                #pass
            count+=1
        final_counts.append(count)
        final_steps.append(steps)
        final_time.append(tm)
    with open("neural_network_counts_taken_diff2cutoff30_mem2.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_counts)+'\n')
    with open("neural_network_steps_taken_diff2cutoff30_mem2.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_steps)+'\n')
    with open("neural_network_time_taken_diff2cutoff30_mem2.txt", "a") as output:
        output.write(str(init_distance)+'\n')
        output.write(str(final_time)+'\n')
        