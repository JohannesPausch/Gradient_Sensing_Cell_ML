from ast import NodeVisitor
from os import read, write
from numpy.core.fromnumeric import _all_dispatcher
from numpy.core.function_base import linspace, logspace
from numpy.lib.function_base import diff
from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
from datawriteread import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import numpy as np
import pickle


######## FIRST CHECK ##################
#direction picking for cell: 
#create some kind of print command for user to select the directions
#direction_sphcoords = pick_direction(0, 5)
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

################### compare particle number accuracy  ###########################
####### THIS WORKS!! EXACT BUT DIFFERENT SEEDS###########
"""
particletest = np.arange(200,401,10)

direction_sphcoords = pick_direction(0,10) #same as sourcenum

Xfinal = np.zeros((10*100,9)) #((sourcenum*seeds,receptornum))
Yfinal = np.zeros((10*100,len(direction_sphcoords)))
for i in particletest:
    for seed_particle in range(1,101): #10 sources corresponding each to the 10 directions -> therefore 10 arrays for each seed of fifo
        X, Y = datacreate(direction_sphcoords, sourcenum=10 ,sourceexact=direction_sphcoords,receptornum=10,particlenum=i, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = X
        Yfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=10 ,receptornum=10,particlenum=i, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile('Xseed_particlenum='+str(i),params, Xfinal)
    write_datafile('Yseed_particlenum='+ str(i),params,Yfinal)
"""
"""
particlenum= np.arange(10,401,10)
instances = np.arange(1,51,1)
direction_sphcoords = pick_direction(0,10) #same as sourcenum
accuracytotal = np.zeros((len(instances),len(particlenum)))
for j in instances: 
    accur =[]
    sc = []
    for i in particlenum:
        X = read_datafile('Xseed_particlenum='+str(i))
        Y = read_datafile('Yseed_particlenum='+str(i))
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
    #print('training data amount percentage:' + str((len(training_x)/len(X))*100))
    #mlp = train(training_x, training_y, layers_tuple = (15), max_iterations=5000,alph=0.01)
        mlp = train(training_x, training_y, layers_tuple =(16), max_iterations=50000, alph=0.001)
    #pickle.dump(mlp, open("MLP_classifier_particles", 'wb'))  
    #restored_clf = pickle.load(open("MLP_classifier", 'rb'))
        acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
        accur.append(acc)
        #sc.append(score)
    accuracytotal[j-1,:]=accur
#write_datafile('accuracy_particle_big18-0.01',params=[0], data=accur)
#write_datafile('score_particle_big18-0.01',params=[0],data=sc)
write_datafile('accuracy_total_particle16-0.001',params=[0],data=accuracytotal)
"""
#changed receptor thing in regular_on_sphere so should give you 9 receptors next time when doing this function

############ RECEPTORS ##################
"""
direction_sphcoords = pick_direction(0,10) #same as sourcenum
#recept= [5,10,20,30,40,50,60,70,80,90,100]
recept = np.arange(5,100,2)
Yfinal = np.zeros((10*100,len(direction_sphcoords)))

for i in recept:
    Xfinal = np.zeros((10*100,len(pick_direction(0,i)))) #((sourcenum*seeds,receptornum))
    for seed_particle in range(1,101): #10 sources corresponding each to the 10 directions -> therefore 10 arrays for each seed of fifo
        X, Y = datacreate(direction_sphcoords, sourcenum=10 ,sourceexact=direction_sphcoords,receptornum=i,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = X
        Yfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=10 ,receptornum=i,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile('Xseed_receptornum='+str(i),params, Xfinal)
    write_datafile('Yseed_receptornum='+ str(i),params,Yfinal)
"""
"""
#recept= [5,10,20,30,40,50,60,70,80,90,100]
recept = np.arange(5,100,2)
instances = np.arange(1,51,1)
direction_sphcoords = pick_direction(0,10) #same as sourcenum
accuracytotal = np.zeros((len(instances),len(recept)))
for j in instances: 
    accur =[]
    sc = []
    for i in recept:
        X = read_datafile('Xseed_receptornum='+str(i))
        Y = read_datafile('Yseed_receptornum='+str(i))
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
        #print('training data amount percentage:' + str((len(training_x)/len(X))*100))
        mlp = train(training_x, training_y, layers_tuple = (i+7), max_iterations=50000,alph=0.001)
        #save_neural_network(mlp, particlenum=i)
        acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
        accur.append(acc)
        sc.append(score)
    accuracytotal[j-1,:]=accur
    #write_datafile('accuracy_recept_big+6-0.01-'+str(j),params=[0], data=accur)
    #write_datafile('score_recept_big+6-0.01-'+str(j),params=[0],data=sc)
write_datafile('accuracy_total_big+7-0.001-iter50000',params=[0],data=accuracytotal)
"""

############ DIFFUSION ##################
"""
direction_sphcoords = pick_direction(0,10) #same as sourcenum
diffusion = np.arange(0.1,2.1,0.1)

Yfinal = np.zeros((10*100,len(direction_sphcoords)))
Xfinal = np.zeros((10*100,9))
for i in diffusion:
    for seed_particle in range(1,101): #10 sources corresponding each to the 10 directions -> therefore 10 arrays for each seed of fifo
        X, Y = datacreate(direction_sphcoords, sourcenum=10 ,sourceexact=direction_sphcoords,receptornum=10,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=i,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = X
        Yfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=10 ,receptornum=10,particlenum=20, recepsurface_ratio=10, distanceexact=3,radiusexact=1,diffusionexact=i,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile('Xseed_diffusion='+str(i),params, Xfinal)
    write_datafile('Yseed_diffusion='+ str(i),params,Yfinal)
"""
"""
diffusion = np.arange(0.1,2.1,0.1)
direction_sphcoords = pick_direction(0,10) 
instances = np.arange(1,51,1)
direction_sphcoords = pick_direction(0,10) #same as sourcenum
accuracytotal = np.zeros((len(instances),len(diffusion)))
for j in instances: 
    accur =[]
    sc =[]
    for i in diffusion:
        X = read_datafile('Xseed_diffusion='+str(i))
        Y = read_datafile('Yseed_diffusion='+str(i))
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
        #print('training data amount percentage:' + str((len(training_x)/len(X))*100))
        mlp = train(training_x, training_y, layers_tuple = (16), max_iterations=50000,alph=0.001)
        #save_neural_network(mlp, particlenum=i)
        acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
        accur.append(acc)
        sc.append(score)
    accuracytotal[j-1,:]=accur
    #write_datafile('accuracy_diffusion_big15-0.01',params=[0], data=accur)
    #write_datafile('score_diffusion_big15-0.01',params=[0],data=sc)
write_datafile('accuracy_total_diffusion_16-0.001',params=[0],data=accuracytotal)
"""
"""

"""
############ CAPTURE AREA ##################### 
#not really needed because same as having more receptors...
#################### DISTANCE ############
direction_sphcoords = pick_direction(0,10) #same as sourcenum
distance= [2,2.5,3,3.5,4,4.5,5,5.5,6]
#6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13]
seednum = 1000 #check more data only for NN check

Yfinal = np.zeros((10*seednum,len(direction_sphcoords)))
Xfinal = np.zeros((10*seednum,9))
for i in distance:
    for seed_particle in range(1,seednum+1): #10 sources corresponding each to the 10 directions -> therefore 10 arrays for each seed of fifo
        X, Y = datacreate(direction_sphcoords, sourcenum=10 ,sourceexact=direction_sphcoords,receptornum=10,particlenum=100, recepsurface_ratio=10, distanceexact=i,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,particle_seed=seed_particle)
        Xfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = X
        Yfinal[10*(seed_particle-1):(10*(seed_particle-1)+ 9),:] = Y
    params = params_string('pick_direction(0,10)', sourcenum=10 ,receptornum=10,particlenum=100, recepsurface_ratio=10, distanceexact=i,radiusexact=1,diffusionexact=1,rateexact=1,receptor_seed=1,initial_source_seed=1,particle_seed=seed_particle)
    write_datafile('Xseed_distance_big='+str(i),params, Xfinal)
    write_datafile('Yseed_distance_big='+ str(i),params,Yfinal)

distance= [2,2.5,3,3.5,4,4.5,5,5.5,6]
#6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13]
direction_sphcoords = pick_direction(0,10) #same as sourcenum
instances = np.arange(1,51,1)
accuracytotal = np.zeros((len(instances),len(distance)))
accuracynn25 = np.zeros((len(instances),len(distance)))
accuracynn35 = np.zeros((len(instances),len(distance)))
accuracynn50 = np.zeros((len(instances),len(distance)))

for j in instances:
    accur =[]
    acn25 = []
    acn35 = []
    acn50 = []

    for i in distance:
        X = read_datafile('Xseed_distance_big='+str(i))
        Y = read_datafile('Yseed_distance_big='+str(i))
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
        mlp = train(training_x, training_y, layers_tuple = (16), max_iterations=50000, alph=0.001)
    #save_neural_network(mlp, particlenum=i)
        pred = predict(mlp, predict_x)
        acc = accuracy_score(predict_y,pred)
        accnn25 = nearest_neighbours_accuracy(direction_sphcoords,predict_y,pred,0.25)
        accnn35 = nearest_neighbours_accuracy(direction_sphcoords,predict_y,pred,0.35)
        accnn50 = nearest_neighbours_accuracy(direction_sphcoords,predict_y,pred,0.50)
        accur.append(acc)
        acn25.append(accnn25)
        acn35.append(accnn35)
        acn50.append(accnn50)
    accuracytotal[j-1,:]=accur
    accuracynn25[j-1,:]=acn25
    accuracynn35[j-1,:]=acn35
    accuracynn50[j-1,:]=acn50

write_datafile('accuracy_total_distances16-0.00150000',params=[0],data=accuracytotal)
write_datafile('accuracynn0.25_total_distances16-0.00150000',params=[0],data=accuracynn25)
write_datafile('accuracynn0.35_total_distances16-0.00150000',params=[0],data=accuracynn35)
write_datafile('accuracynn0.50_total_distances16-0.00150000',params=[0],data=accuracynn50)

"""
"""

############# NEURAL NETWORK ################
"""
direction_sphcoords = pick_direction(0,10) #same as sourcenum
alpha= [0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0]
instances = np.arange(1,51,1)
accuracytotal = np.zeros((len(instances),len(alpha)))
for j in instances:
    accur =[]
    for i in alpha:
        X = read_datafile('Xseed_distanceonly=2')
        Y = read_datafile('Yseed_distanceonly=2')
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
    #print('training data amount percentage:' + str((len(training_x)/len(X))*100))
        mlp = train(training_x, training_y, layers_tuple = (16), max_iterations=10000, alph=i, solve ='adam')
    #save_neural_network(mlp, particlenum=i)
        acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
        accur.append(acc)
    accuracytotal[j-1,:]=accur
write_datafile('accuracy_total_alphas_16',params=[0],data=accuracytotal)
"""
"""
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
parameters ={'solver': ['lbfgs'],'alpha':[0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0], 'hidden_layer_sizes' : np.arange(10,20),'early_stopping':[True]}
#parameters = {'solver': ['adam'], 'max_iter': [1000,2000,3000,4000,5000 ], \
    #'alpha':np.logspace(-1, 1, 5), \
    #'hidden_layer_sizes':np.arange(15, 25), 'beta_1':np.linspace(0.1, 0.9, 5), \
    #'beta_2' : np.linspace(0.1, 0.999, 5), 'epsilon': np.logspace(-8,-2,5),  'early_stopping':[True]}
clf = GridSearchCV(MLPClassifier(), parameters)
X = read_datafile('Xseed_distance=2') #distanceonly when using 'adam' (bottom results)
Y = read_datafile('Yseed_distance=2')
X = MinMaxScaler().fit_transform(X)
clf.fit(X, Y)
report(clf.cv_results_)
"""
"""
Model with rank: 1
Mean validation score: 0.354 (std: 0.308)
Parameters: {'alpha': 0.1, 'early_stopping': True, 'hidden_layer_sizes': 20, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.227 (std: 0.254)
Parameters: {'alpha': 0.1, 'early_stopping': True, 'hidden_layer_sizes': 21, 'solver': 'adam'}

Model with rank: 3
Mean validation score: 0.222 (std: 0.254)
Parameters: {'alpha': 0.1, 'early_stopping': True, 'hidden_layer_sizes': 18, 'solver': 'adam'}

Model with rank: 1
Mean validation score: 0.481 (std: 0.307)
Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': 19, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.480 (std: 0.308)
Parameters: {'alpha': 0.01, 'early_stopping': True, 'hidden_layer_sizes': 15, 'solver': 'adam'}

Model with rank: 3
Mean validation score: 0.477 (std: 0.297)
Parameters: {'alpha': 0.001, 'early_stopping': True, 'hidden_layer_sizes': 18, 'solver': 'adam'}
"""
"""
plt.rcParams.update({'font.size': 18})
nodes= range(3,50)
#accuracy= [0.052, 0.34, 0.44, 0.628, 0.088, 0.692, 0.728, 0.676, 0.752, 0.772, 0.716, 0.764, 0.74, 0.648, 0.792, 0.768, 0.768, 0.74, 0.72, 0.732, 0.752, 0.756, 0.744, 0.748, 0.728, 0.752, 0.804, 0.76, 0.728, 0.748, 0.716, 0.72, 0.728, 0.688, 0.684, 0.736, 0.74, 0.768, 0.712, 0.74, 0.748, 0.7, 0.712, 0.716, 0.752, 0.74, 0.692, 0.724, 0.724, 0.704, 0.712, 0.716, 0.74, 0.724, 0.72, 0.712, 0.704, 0.728, 0.712, 0.688, 0.728, 0.72, 0.696, 0.656, 0.732, 0.696, 0.72, 0.732, 0.708, 0.74, 0.744, 0.668, 0.7, 0.732, 0.712, 0.672, 0.76, 0.72, 0.724, 0.74, 0.712, 0.728, 0.748, 0.732, 0.74, 0.744, 0.736, 0.728, 0.708, 0.724, 0.712, 0.74, 0.7, 0.796, 0.712, 0.7, 0.78, 0.712, 0.724, 0.724, 0.724, 0.748, 0.684, 0.7, 0.756, 0.7, 0.732, 0.732, 0.696, 0.764, 0.76, 0.728, 0.728, 0.784, 0.716, 0.764, 0.736, 0.708, 0.74, 0.744, 0.732, 0.784, 0.68, 0.716, 0.744, 0.748, 0.724, 0.76, 0.724, 0.74, 0.724, 0.748, 0.736, 0.704, 0.768, 0.764, 0.712, 0.732, 0.72, 0.7, 0.764, 0.74, 0.76, 0.756, 0.724, 0.736, 0.74, 0.724, 0.708, 0.716, 0.696, 0.72, 0.764, 0.704, 0.688, 0.712, 0.74, 0.804, 0.696, 0.716, 0.696, 0.748, 0.708, 0.776, 0.716, 0.756, 0.724, 0.74, 0.752, 0.736, 0.724, 0.72, 0.756, 0.712, 0.74, 0.72, 0.72, 0.74, 0.696, 0.752, 0.768, 0.708, 0.68, 0.76, 0.728, 0.72, 0.74, 0.732, 0.796, 0.704, 0.676, 0.756, 0.756, 0.696, 0.728, 0.744, 0.74]
accuracy= [0.5, 0.3276, 0.6968, 0.702, 0.7304, 0.7356, 0.7096, 0.7716, 0.7812, 0.7688, 0.7656, 0.7808, 0.7716, 0.7644, 0.764, 0.7988, 0.7776, 0.78, 0.794, 0.7772, 0.786, 0.7996, 0.7952, 0.7748, 0.7928, 0.782, 0.7876, 0.7788, 0.7844, 0.7932, 0.792, 0.796, 0.788, 0.8008, 0.7884, 0.792, 0.7772, 0.7768, 0.7884, 0.794, 0.7888, 0.794, 0.7756, 0.7972, 0.79, 0.7916, 0.7948]
write_datafile('nodesaccuracy',params=[0] ,data=accuracy,linewidth = 0, marker = 'o', color = 'blue')
plt.figure(figsize=(8,6))
plt.plot(nodes,accuracy, '--o',)
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Nodes of first hidden layer')
plt.savefig('nodes2.png')

"""









################# BEFORE ######################

