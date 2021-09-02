import matplotlib as plt
import numpy as np
from scipy.sparse import data
from IdealDirection import *
from datawriteread import *
from ReceptorNeuralNetwork import *
plt.rcParams.update({'font.size': 30})

#particlenum= [5,10,20,30,40,50,60,70,80,90,100]
particlenum= np.arange(10,401,10)
accuracy = read_datafile('accuracy_total_particle16-0.001')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(13,8))
plt.errorbar(particlenum,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o',markersize =11, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of particles')
plt.savefig('particles16-0.001.png')


recept = np.arange(5,100,2)
plt.figure(figsize=(13,8))
accuracy = read_datafile('accuracy_total_big+7-0.001-iter50000')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
indx = np.where(max(accmean)==accmean)
print(indx)
print(accmean[indx])
print(recept[indx[0]])
err= np.std(accuracy, axis=0)
plt.errorbar(recept,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o',markersize =11, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of receptors')
plt.savefig('recept+7-0.001_many.png')

diffusion = np.arange(0.1,2.1,0.1)
accuracy = read_datafile('accuracy_total_diffusion_16-0.001')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(13,8))
plt.errorbar(diffusion,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o', markersize =14, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Diffusion constant')
plt.savefig('diffusionerr16001.png')

distance= [2,3,4,5,6,7,8,9,10]
accuracy = read_datafile('accuracy_total_distances16-0.0015000')
plt.figure(figsize=(10,10))
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(13,8))
plt.errorbar(distance,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o', markersize =14, color = 'red')
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
plt.errorbar(nodes,accmean,yerr=err,elinewidth=2, linewidth = 0, marker = 'o', markersize= 14, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of nodes in hidden layer')
plt.savefig('nodeserr.png')

alpha= [0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0]
accuracy = read_datafile('accuracy_total_alphas_16')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(13,8))
plt.errorbar(alpha,accmean,yerr=err,elinewidth=1, linewidth = 0, marker = 'o', markersize= 8, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel(r'$\alpha$')
plt.savefig('alpha.png')