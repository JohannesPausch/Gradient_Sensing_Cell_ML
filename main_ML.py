from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
import numpy as np

######## data ##################
#direction picking for cell: 
#create some kind of print command for user to select the directions
direction_sphcoords = pick_direction(0, 10)

#create data
X, Y = datacreate(direction_sphcoords, receptornum=20,particlenum=10, recepsurface_ratio=20,diffusionexact=1,distanceexact=1.5,radiusexact=1,rateexact=1)
print(X)
print(Y)
"""
###### train and test ##########
training_x, training_y, predict_x, predict_y = separate_train_set(X, Y)
mlp = train(training_x, training_y, layers_tuple = (6,4), max_iterations=1000)

filename = save_neural_network(mlp, distance=10,rate=None,diffusion=None,seed=None,cutoff=None,events=None,iterations=None)
restored_mlp = load_neural_network(filename)
print(filename)
    
test(restored_mlp, predict_x, predict_y))
"""