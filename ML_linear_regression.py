#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
import scipy.linalg
 
#from analyticalreg import *
#from reshapevalues import *
#from random_3d_rotation import *

############## DATA origin source ###########################
data = np.loadtxt('/Users/Paula/Gradient_Sensing_Cell_ML-main/BrownianParticle_ref.txt',
				delimiter=' ', 	# String used to separate values
				usecols=[0,1,2,3,4,5,6,7], 	# Specify which columns to read
				dtype=np.double) 		# The type of the resulting array
theta_data=data[:,2]
phi_data=data[:,3]
numtrain=len(theta_data)+1

Ydata=np.zeros((1,2)) #origin source
Xdata=np.transpose(np.concatenate((np.transpose(theta_data).reshape(1,numtrain-1),np.transpose(phi_data).reshape(1,numtrain-1))))
####################FUNCTIONS###################################
  #Spyder still doesn't let me use modules in the editor
def analyticalreg(x,y): 
    #method to solve for beta analytically
    #(X^tX)^-1X^ty
    
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)

def random_3d_rotation(theta,phi,use_seed=None):   
    try:
        np.random.seed(seed=use_seed)
        y=np.matmul(special_ortho_group.rvs(3),np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))#special_ortho_group.rvs(3)
        return np.arccos(y[2]), np.mod(np.arctan2(y[1],y[0]),2*np.pi)
    except ValueError:
        if phi.shape != theta.shape:  print("Error in random_3d_rotation: Shapes of phi and theta don't match")
        else: print("Unknown error in random_3d_rotation")
        return theta,phi


def reshapevalues(theta,phi):
    result = np.empty((2*theta.size), dtype=theta.dtype) 
    result[0::2]=theta
    result[1::2]=phi
    return result

##################### SET UP VARIABLES  ######################

# A sweep is done with the window of size "groupsize" that goes down the rows of matrix Xdata with step 1 row.

# Xtot is a matrix of size: (number of groups) x (2*groupsize+1). The 2*groupsize+1 is the column dimension since 
# each row takes all the data of a group and organizes it in the following way using the reshapevalues function: 
# theta_1 phi_1 theta_2 phi_2... theta(2*groupsize) phi_(2*groupsize). In the end a 1 is added as the first variable of each row
# to account for the intercept.

# In Xtot, the row 0 is not rotated and row 1 is the random rotation of row 0,
# the rest of the rows correspond to randomly rotated variables of the selected window of Xdata.

groupsize=40

for i in range(0,numtrain-1):
    if (i+groupsize>numtrain-1):
        print(i+groupsize) #The window is larger than the actual rows left of the training matrix. 
        break
    else:
        theta=Xdata[i:i+groupsize,0]
        phi=Xdata[i:i+groupsize,1]
        Yrotdata=np.array(random_3d_rotation(Ydata[0,0],Ydata[0,1])).reshape(1,2)
        [rtheta,rphi]=random_3d_rotation(theta,phi)
        Xrotdata=reshapevalues(rtheta,rphi).reshape(1,2*groupsize)
        if i==0:
            Xinit=reshapevalues(theta,phi).reshape(1,2*groupsize)
            Xtot=np.concatenate((Xinit,Xrotdata))
            Ytot= np.concatenate((Ydata, Yrotdata))
        else: 
            Xtot=np.concatenate((Xtot,Xrotdata))
            Ytot= np.concatenate((Ytot, Yrotdata))
        
#add intercept: 
Xtot=np.insert(Xtot, 0, 1, axis=1)
##################### Linear regression ############################################


beta= analyticalreg(Xtot, Ytot)


##########################VISUALS#############################################

