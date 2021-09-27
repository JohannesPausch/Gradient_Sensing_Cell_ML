from mpl_toolkits import mplot3d
from datawriteread import *
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
filename = 'coordinates20'
fig = plt.figure()
ax = plt.axes(projection='3d')
data = np.array(read_datafile(filename))

ax.scatter(data[:,1], data[:,2], data[:,3])
ax.scatter(data[len(data[:,0]),1],data[len(data[:,0]),1],data[len(data[:,0]),1],s=50) 
ax.set_xlabel('x source')
ax.set_ylabel('y source')
ax.set_zlabel('z source')
ax.set_xlim3d(-6, 6)
ax.set_ylim3d(-6, 6)
ax.set_zlim3d(-6, 6)
plt.show()
