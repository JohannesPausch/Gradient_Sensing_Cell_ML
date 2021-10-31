import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 

import sys
sys.path.append('/Users/taliarahall/Gradient_Sensing_Cell_ML/Code')

from datawriteread import read_datafile

neuraldata = read_datafile('neural_network_steps_data')
greedydata= read_datafile('greedy_algorithm_steps_data')

dist = np.arange(2,15,1)

mean_neural =[]
std_neural =[]

for i in neuraldata:
    mean_neural.append(np.mean(i))
    std_neural.append(np.std(i))

mean_greedy =[]
std_greedy =[]

for j in greedydata:
    mean_greedy.append(np.mean(j))
    std_greedy.append(np.std(j))


plt.plot(dist, mean_neural, color='brown', label='Neural Network')
plt.plot(dist, mean_greedy, color='green', label='Greedy Algorithm')
plt.errorbar(dist, mean_neural, yerr=std_neural, color='brown',fmt='o',  markersize=6, capsize=5)
plt.errorbar(dist, mean_greedy, yerr=std_greedy, color='green',fmt='o',  markersize=6, capsize=5)
plt.legend()
plt.xlabel('Distance of Cell from Source')
plt.ylabel('Steps to find source')
plt.grid()
plt.show()