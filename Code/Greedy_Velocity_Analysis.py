import matplotlib.pyplot as plt
import numpy as np 


distance_5_v_01 = np.loadtxt('greedy_algorithm_failed_5_talia.dat',unpack=True)
distance_15_v_01 = np.loadtxt('greedy_algorithm_failed_15_talia.dat',unpack=True)

cutoffs = np.arange(20,52,2)

init_distances = [5,15]

plt.plot(cutoffs, distance_5_v_01[1],label='Distance=5, Velocity=0.01')
plt.plot(cutoffs, distance_15_v_01[1],label='Distance=15, Velocity=0.01')

plt.xlabel('Cutoff')
plt.ylabel('Number of Failures')


distance_10_v_055 = np.loadtxt('greedy_algorithm_failed_10_v0.055.dat',unpack=True)
distance_10_v_055_2 = np.loadtxt('greedy_algorithm_failed_10_v0.055_2.dat',unpack=True)

distance_5_v_055 = np.loadtxt('greedy_algorithm_failed_5_v0.055.dat',unpack=True)
distance_5_v_055_2 = np.loadtxt('greedy_algorithm_failed_5_v0.055_2.dat',unpack=True)

distance_15_v_055 = np.loadtxt('greedy_algorithm_failed_15_v0.055_2.dat',unpack=True)

plt.errorbar(cutoffs, np.mean([distance_10_v_055[1],distance_10_v_055_2[1]],axis=0), yerr=np.std([distance_10_v_055[1],distance_10_v_055_2[1]],axis=0), label='Distance=10, Velocity=0.055')
plt.errorbar(cutoffs, np.mean([distance_5_v_055[1],distance_5_v_055_2[1]],axis=0), yerr=np.std([distance_5_v_055[1],distance_5_v_055_2[1]],axis=0), label='Distance=5, Velocity=0.055')
plt.plot(cutoffs, distance_15_v_055[1], label='Distance=15, Velocity=0.055')
plt.legend()
plt.show()
