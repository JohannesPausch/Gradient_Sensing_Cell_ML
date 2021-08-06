from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
import numpy as np

######## data ##################
#direction picking for cell: 
direction_sphcoords = pick_direction(0, 20)

#create data
X, Y = datacreate(direction_sphcoords)


###### train and test ##########
training_x, training_y, predict_x, predict_y = separate_train_set(X, Y)
mlp = train(training_x, training_y, layers_tuple = (6,4), max_iterations=1000)

save_neural_network(mlp)
restored_mlp = load_neural_network()
test(restored_mlp, predict_x, predict_y)