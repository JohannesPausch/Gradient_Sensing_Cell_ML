import matplotlib as plt
import numpy as np
from scipy.sparse import data
from IdealDirection import *
from datawriteread import *
from ReceptorNeuralNetwork import *

#particlenum= [5,10,20,30,40,50,60,70,80,90,100]
particlenum= np.arange(10,200,10)
accuracy = read_datafile('accuracy_particle_big15-0.01')
plt.figure(figsize=(8,6))
plt.plot(particlenum,accuracy[1],'--o')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of particles')
plt.savefig('particles15-0.01.png')

plt.rcParams.update({'font.size': 30})

recept = np.arange(5,100,2)
plt.figure(figsize=(11,8))
accuracy = read_datafile('accuracy_total_big+6-0.01')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.errorbar(recept,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o',markersize =11, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of receptors')
plt.savefig('recept+9-0.01_many.png')

diffusion = np.arange(0.1,2.1,0.1)
accuracy = read_datafile('accuracy_total_diffusion_15-0.01')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(11,9))
plt.errorbar(diffusion,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o', markersize =14, color = 'red')

plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Diffusion constant')
plt.savefig('diffusionerr.png')

distance= [2,3,4,5,6,7,8,9,10]
accuracy = read_datafile('accuracy_distance')
plt.figure(figsize=(10,10))
plt.plot(distance,accuracy[1],'--o')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Distance from source')
plt.savefig('distance.png')

nodes = nodes= range(3,40)
accuracy = read_datafile('accuracy_total_nodes')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(13,8))
#plt.figure()
plt.errorbar(nodes,accmean,yerr=err,elinewidth=1, linewidth = 0, marker = 'o', markersize= 8, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of nodes in hidden layer')
plt.savefig('nodeserr.png')