from os import read
from numpy.core.function_base import linspace
from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
from datawriteread import *
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


######## data ##################
#direction picking for cell: 
#create some kind of print command for user to select the directions
#direction_sphcoords = pick_direction(0, 10)
#print(direction_sphcoords)

#why index 0 more activated than the rest?
#X, Y = datacreate(direction_sphcoords, sourcenum=5 ,receptornum=10,particlenum=10, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionnum=3,ratenum=2,receptor_seed=1,initial_source_seed=1)
#params = params_string('pick_direction(0,10)', sourcenum=5 ,receptornum=10,particlenum=10, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionnum=3,ratenum=2,receptor_seed=1,initial_source_seed=1)
#write_datafile('X_test=',params, X)
#write_datafile('Y_test=',params, Y)
#check double index in Y
#coords = cart2spherical_point(-4.651031627755137,2.903085818191935,-0.43588706087042833)
#print(direction_sphcoords)
#print(coords)
#ind = ideal_direction(coords[1],coords[2],direction_sphcoords,1)
#print(ind)
#check 
"""
################### compare particle number accuracy ###########################
particletest= [1,5,10,15,20]
#same source directions for each data: same Y
#5*3*3*3*2=270 data arrays to test

for i in particletest:
    X, Y = datacreate(direction_sphcoords, sourcenum=5 ,receptornum=10,particlenum=i, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionnum=3,ratenum=2,receptor_seed=1,initial_source_seed=1)
    params = params_string('pick_direction(0,10)', sourcenum=5 ,receptornum=10,particlenum=i, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionnum=3,ratenum=2,receptor_seed=1,initial_source_seed=1)
    write_datafile('X_particlenum='+str(i),params, X)
    write_datafile('Y_particlenum='+str(i),params, Y)
"""
# TRAIN #
particletest= [1,5,10,15,20]
accuracy =[]
random.seed(1)
for i in particletest:
    print(i)
    X = read_datafile('X_particlenum='+str(i))
    Y = read_datafile('Y_particlenum='+str(i))
    training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
    print('training data amount percentage:' + str((len(training_x)/len(X))*100))
    mlp = train(training_x, training_y, layers_tuple = (100,50,25), max_iterations=1000)
    save_neural_network(mlp, particlenum=i)
    acc, probs, score = test(mlp, predict_x, predict_y)
    accuracy.append(acc)
print(accuracy)

"""
################### compare receptor number accuracy ###########################
receptest= [10,20,30,40,50]
#same source directions for each data: same Y
loops = 5*3*3*3*2 #270 data arrays to test
print(loops)

for i in receptest:
    X, Y = datacreate(direction_sphcoords, sourcenum=5 ,receptornum=i,particlenum=20, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionnum=3,ratenum=2,receptor_seed=1,initial_source_seed=1)
    params = params_string('pick_direction(0,10)', sourcenum=5 ,receptornum=i,particlenum=20, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionnum=3,ratenum=2,receptor_seed=1,initial_source_seed=1)
    write_datafile('X_receptornum='+str(i),params, X)
    write_datafile('Y_receptornum='+str(i),params, Y)
    """
receptest= [10,20,30,40,50]
accuracy =[]
random.seed(1)
for i in receptest:
    print(i)
    X = read_datafile('X_receptornum='+str(i))
    Y = read_datafile('Y_receptornum='+str(i))
    training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
    print('training data amount percentage:' + str((len(training_x)/len(X))*100))
    mlp = train(training_x, training_y, layers_tuple = (100,50,25), max_iterations=1000)
    save_neural_network(mlp, particlenum=i)
    acc, probs, score = test(mlp, predict_x, predict_y)
    accuracy.append(acc)
print(accuracy)
################### compare diffusion constants accuracy ###########################
"""
diffusion_constants  = np.linspace(0.5,1,5)
#same source directions for each data: same Y
loops = 5*3*3*3*2 #270 data arrays to test
print(loops)

for i in diffusion_constants:
    X, Y = datacreate(direction_sphcoords, sourcenum=5 ,receptornum=10,particlenum=20, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionexact=i,ratenum=2,receptor_seed=1,initial_source_seed=1)
    params = params_string('pick_direction(0,10)', sourcenum=5 ,receptornum=10,particlenum=20, recepsurface_ratio=10, distancenum=3,radiusnum=3,diffusionexact=i,ratenum=2,receptor_seed=1,initial_source_seed=1)
    write_datafile('X_diffusion='+str(i),params, X)
    write_datafile('Y_diffusion='+str(i),params, Y)
#compare layer nodes accuracy 
"""
diffusion_constants  = np.linspace(0.5,1,5)
accuracy =[]
random.seed(1)
for i in diffusion_constants:
    print(i)
    X = read_datafile('X_diffusion='+str(i))
    Y = read_datafile('Y_diffusion='+str(i))
    training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
    print('training data amount percentage:' + str((len(training_x)/len(X))*100))
    mlp = train(training_x, training_y, layers_tuple = (100,50,25), max_iterations=1000)
    save_neural_network(mlp, particlenum=i)
    acc, probs, score = test(mlp, predict_x, predict_y)
    accuracy.append(acc)
print(accuracy)

Ytest = read_datafile('Y_diffusion=0.5')