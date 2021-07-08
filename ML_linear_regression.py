import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
import scipy.linalg
 
#from analytical import *
#from CostFunction import *
#from GradDescent import *
#from random_3d_rotation import *
############## DATA origin source ###########################
data = np.loadtxt('/Users/Paula/Gradient_Sensing_Cell_ML-main/BrownianParticle_ref.txt',
				delimiter=' ', 	# String used to separate values
				usecols=[0,1,2,3,4,5,6,7], 	# Specify which columns to read
				dtype=np.double) 		# The type of the resulting array
event=data[:,0]
iteration=data[:,1]
theta=data[:,2]
phi=data[:,3]
x=data[:,4]
y=data[:,5]
z=data[:,6]
t=data[:,7]
numtrain=len(x)+1


ytheta=source_theta_rel_sphere=np.zeros(numtrain-1) # theta of source
yphi=source_phi_rel_sphere=np.zeros(numtrain-1)# phi of source, doesn't matter when source in the x axis

####################FUNCTIONS###################################
  
def analyticalreg(x,y): 
    #method to solve for beta analytically
    #(X^tX)^-1X^ty
    x=np.transpose(x)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)

def random_3d_rotation(phi,theta,use_seed=None):  #modified  
    try:
        np.random.seed(seed=use_seed)
        y=np.matmul(special_ortho_group.rvs(3),np.array([np.cos(theta),np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)]))#special_ortho_group.rvs(3)
        return np.arccos(y[0]), np.mod(np.arctan2(y[2],y[1]),2*np.pi)
    except ValueError:
        if phi.shape != theta.shape:  print("Error in random_3d_rotation: Shapes of phi and theta don't match")
        else: print("Unknown error in random_3d_rotation")
        return phi,theta
    
def addsources(theta,phi,ytheta,yphi,nsources):
    lentrain=len(theta)
    origintheta=theta
    originphi=phi
    for i in range(1,nsources):
        
        [rytheta,ryphi]=random_3d_rotation(0,0,i)
        rytheta=rytheta*np.ones(lentrain)
        ryphi=ryphi*np.ones(lentrain)

        [rtheta,rphi]=random_3d_rotation(origintheta,originphi,i)
   
        ytheta=np.concatenate((ytheta,rytheta))
        yphi=np.concatenate((yphi,ryphi))
        theta=np.concatenate((theta,rtheta))
        phi=np.concatenate((phi,rphi))
    return [theta,phi,ytheta,yphi]

def factors(x):
   f=[0]
   for i in range(1, x + 1):
       if x % i == 0 and i % 3 ==0 :
          f.append(i)
   return f

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
    
#https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/


nsources=6
[theta,phi,ytheta,yphi]=addsources(theta,phi,ytheta,yphi,nsources)

##################### SET UP VARIABLES  ######################
groupsizeapprox=30
totdat=len(theta)
groupsize=closest(factors(totdat),groupsizeapprox)


X=np.concatenate((np.ones(nsources*(numtrain-1)).reshape(1,nsources*(numtrain-1)),np.transpose(theta).reshape(1,nsources*(numtrain-1)),np.transpose(phi).reshape(1,nsources*(numtrain-1))))
                 ## include steps?
X1= np.zeros(3*totdat).reshape(groupsize,int(3*totdat/groupsize))
for j in range(0,totdat):
    for k in range(0,int(3*totdat/groupsize)):
        for i in np.array(range(0,groupsize))[::3]:
            X1[i:i+3,k]=X[:,j]

#X1=X.reshape(groupsize,int(3*totdat/groupsize)) doesn't work

Y = (np.concatenate((ytheta,yphi)).reshape(2,nsources*(numtrain-1))).T

Y1= np.zeros(3*totdat).reshape(groupsize,int(3*totdat/groupsize))
for j in range(0,totdat):
    for k in range(0,int(3*totdat/groupsize)):
        for i in np.array(range(0,groupsize))[::2]:
            Y1[i:i+2,k]=Y[j,:]
            
beta = np.random.rand(3,2)  #weights
#######################RUN ITERATIONS(Gradient descent)###############################################
"""
numiter = 1000
beta = np.random.rand(4,1)  #weights for theta
alpha = 1e-12#learning rate
J=np.zeros((numiter,1))
for i in range(0,numiter): #repeat until convergence
    print(beta)
    beta= GradDescent(alpha,beta,X,y1)
    J[i,0]= CostFunction(beta,X,y1)
""" 
##################### regression ############################################

beta= analyticalreg(X, Y)
##########################VISUALS#############################################

