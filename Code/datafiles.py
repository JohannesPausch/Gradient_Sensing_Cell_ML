from ast import NodeVisitor
from os import read, write
from numpy.core.fromnumeric import _all_dispatcher
from numpy.core.function_base import linspace, logspace
from numpy.lib.function_base import diff
from ReceptorNeuralNetwork import *
from datacreate import *
from IdealDirection import *
from datawriteread import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

direction_sphcoords = pick_direction(0,10)
directions = len(direction_sphcoords)

receptornumber = 10 #same as sourcenum for this simulation
actual_receptornum = len(pick_direction(0,receptornumber)) 

sources = 10
seeds = 100


################### particles  ###########################
particletest = np.arange(10,401,10)
filenameX= 'Xparticles='
filenameY='Yparticles='


Xfinal = np.zeros((sources*seeds,actual_receptornum)) #((sourcenum*seeds,receptornum))
Yfinal = np.zeros((sources*seeds,directions))
for i in particletest:
    for seed_particle in range(1,seeds+1): #100 instances per source point

       #use datacreate with exact parameters except the one that is being tested
        X, Y = datacreate(direction_sphcoords, sourcenum=sources ,sourceexact=direction_sphcoords,receptornum=receptornumber,particlenum=i, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
       
        Xfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = X
        Yfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = Y

    params = params_string('pick_direction(0,10)', sourcenum=sources ,receptornum=receptornumber,particlenum=i, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile(filenameX+str(i),params, Xfinal)
    write_datafile(filenameY+str(i),params,Yfinal)

# a bit repetitive for clarity ################### receptors  ###########################

recept = np.arange(5,100,2)
filenameX = 'Xrecept='
filenameY = 'Yrecept='

Yfinal = np.zeros((sources*seeds,len(direction_sphcoords)))
receptors = []
for i in recept:
    Xfinal = np.zeros((sources*seeds,len(pick_direction(0,i)))) #((sourcenum*seeds,receptornum))
    receptors= receptors.append(len(pick_direction(0,i)))
    for seed_particle in range(1,seeds+1): 
        X, Y = datacreate(direction_sphcoords, sourcenum=sources ,sourceexact=direction_sphcoords,receptornum=i,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = X
        Yfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=sources ,receptornum=i,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile(filenameX+str(i),params, Xfinal)
    write_datafile(filenameY+ str(i),params,Yfinal)
write_datafile('receptors',params=[0],data=receptors)

################### diffusion  ###########################

diffusion = np.arange(0.1,2.1,0.1)
filenameX = 'Xdiffusion='
filenameY = 'Ydiffusion='

Xfinal = np.zeros((sources*seeds,actual_receptornum))
Yfinal = np.zeros((sources*seeds,len(direction_sphcoords)))
for i in diffusion:
    for seed_particle in range(1,seeds+1): 
        X, Y = datacreate(direction_sphcoords, sourcenum=sources ,sourceexact=direction_sphcoords,receptornum=receptornumber,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=i,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = X
        Yfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=sources ,receptornum=receptornumber,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=i,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile(filenameX+str(i),params, Xfinal)
    write_datafile(filenameY+ str(i),params,Yfinal)

############ distances #################
direction_sphcoords = pick_direction(0,10) #same as sourcenum #10
distance= [2,3,4,5,6,7,8,9,10]
filenameX = 'Xseed_distance='
filenameY = 'Yseed_distance='

Yfinal = np.zeros((sources*seeds,len(direction_sphcoords)))
Xfinal = np.zeros((sources*seeds,actual_receptornum)) 
for i in distance:
    for seed_particle in range(1,seeds+1): #10 sources corresponding each to the 10 directions -> therefore 10 arrays for each seed of fifo
        X, Y = datacreate(direction_sphcoords, sourcenum=sources ,sourceexact=direction_sphcoords,receptornum=receptornumber,particlenum=20, recepsurface_ratio=10, distanceexact=i,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = X
        Yfinal[sources*(seed_particle-1):(sources*(seed_particle-1)+ sources-1),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=sources ,receptornum=receptornumber,particlenum=20, recepsurface_ratio=10, distanceexact=i,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile(filenameX+str(i),params, Xfinal)
    write_datafile(filenameY+ str(i),params,Yfinal)


################### distances for NN parameter check and total NN  ###########################
distance= [2,2.5,3,3.5,4,4.5,5,5.5,6]
seeds = 1000 #check more data only for NN check
filenameX = 'Xseed_distance_seeds1000particles100='
filenameY = 'Yseed_distance_seeds1000particles100='


Yfinal = np.zeros((sources*seeds,len(direction_sphcoords)))
Xfinal = np.zeros((sources*seeds,actual_receptornum)) 
for i in distance:
    for seed_particle in range(1,seeds+1): 
        X, Y = datacreate(direction_sphcoords, sourcenum=sources ,sourceexact=direction_sphcoords,receptornum=receptornumber,particlenum=100, recepsurface_ratio=10, distanceexact=i,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[sources*(seed_particle-1):(10*(seed_particle-1)+ sources-1),:] = X 
        Yfinal[sources*(seed_particle-1):(10*(seed_particle-1)+ sources-1),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=sources ,receptornum=receptornumber,particlenum=100, recepsurface_ratio=10, distanceexact=i,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile(filenameX+str(i),params, Xfinal)
    write_datafile(filenameY+ str(i),params,Yfinal)



