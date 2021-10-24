from datawriteread import *
from ReceptorNeuralNetwork import *
from IdealDirection import *
from sklearn.preprocessing import MinMaxScaler

distance= [2,2.5,3,3.5,4,4.5,5,5.5,6]
direction_sphcoords = pick_direction(0,10) #same as sourcenum
#instances = np.arange(1,51,1)
#X = read_datafile('Xtseed_distanceonly=2')
#Xfinal = np.zeros((len(distance)*len(X),len(X[0,:])))
Xfinal = read_datafile('X_total_distances_update')
#Yfinal = np.zeros((len(distance)*len(X),len(direction_sphcoords)))
Yfinal = read_datafile('Y_total_distances_update')
     

training_x, predict_x,training_y, predict_y = train_test_split(Xfinal, Yfinal)
mlp = train(training_x, training_y, layers_tuple = (19), max_iterations=50000, alph=0.1,solve='lbfgs')
acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
print(acc)
save_neural_network(mlp, 'Total_mlp2')

write_datafile('X_total_distances_update',params=[0],data=Xfinal)
write_datafile('Y_total_distances_update',params=[0],data=Yfinal)
