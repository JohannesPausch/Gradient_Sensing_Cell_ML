import matplotlib as plt
import numpy as np
from scipy.sparse import data
from IdealDirection import *
from datawriteread import *
from ReceptorNeuralNetwork import *

particlenum= [5,10,20,30,40,50,60,70,80,90,100]
accuracy = read_datafile('accuracy_particle')
plt.figure(figsize=(8,6))
plt.plot(particlenum,accuracy[1],'--o')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of particles')
plt.savefig('particles.png')

recept= [5,10,20,30,40,50,60,70,80,90,100]
accuracy = read_datafile('accuracy_recept')
plt.figure(figsize=(8,6))
plt.plot(recept,accuracy[1],'--o')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of receptors')
plt.savefig('recept.png')

diffusion = [0.1,0.25,0.5,0.75,1]
accuracy = read_datafile('accuracy_diffusion')
plt.figure(figsize=(8,6))
plt.plot(diffusion,accuracy[1],'--o')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Diffusion constant')
plt.savefig('diffusion.png')

distance= [2,3,4,5,6,7,8,9,10]
accuracy = read_datafile('accuracy_distance')
plt.figure(figsize=(8,6))
plt.plot(distance,accuracy[1],'--o')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Distance from source')
plt.savefig('distance.png')