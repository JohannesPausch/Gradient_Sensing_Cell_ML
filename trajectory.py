from mpl_toolkits import mplot3d
from datawriteread import *
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
filename = 'coordinates4_cart'
fig = plt.figure()
ax = plt.axes(projection='3d')
data = np.array(read_datafile(filename))
print(data)
ax.scatter3D(data[:,1], data[:,2], data[:,3])
ax.scatter3D(data[0,1],data[0,2],data[0,3],'r') 

plt.show()
