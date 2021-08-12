from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
import numpy as np
import pickle

######## data ##################
#direction picking for cell: 
#create some kind of print command for user to select the directions
direction_sphcoords = pick_direction(0, 10)

#create data
X, Y = datacreate(direction_sphcoords, receptornum=10,particlenum=20, recepsurface_ratio=10)
print(X)
print(Y)
pickle.dump(X, open(X, 'wb'))
pickle.dump(X, open(X, 'wb'))
def save_X(X, distance=None,rate=None,diffusion=None):
    filename = 'X'
    if distance != None:
        filename += ' -d '+str(distance)
    if rate != None:
        filename += ' -r '+str(rate)
    if diffusion != None:
        filename += ' -s '+str(diffusion)
    pickle.dump(X, open(filename, 'wb'))
    return filename
save_X(X)
save_X(Y,distance=1)
"""
###### train and test ##########
training_x, training_y, predict_x, predict_y = separate_train_set(X, Y)
mlp = train(training_x, training_y, layers_tuple = (6,4), max_iterations=1000)

filename = save_neural_network(mlp, distance=10,rate=None,diffusion=None,seed=None,cutoff=None,events=None,iterations=None)
restored_mlp = load_neural_network(filename)
print(filename)
    
test(restored_mlp, predict_x, predict_y)
"""